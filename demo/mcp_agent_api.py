"""
MCP Agent API Module
VLM Agent 로직을 분리한 모듈

Note: 해당 파일을 실제 백엔드에 적용 가능하나, 서비스의 추론 속도를 올리기 위해 현 프로젝트에서는 modis_server_standalone.py를 사용함.
"""

import asyncio
import json
import os
import sys
import re
import logging
from typing import Optional

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. 설정
# -----------------------------------------------------------------------------
SGLANG_API_URL = "http://localhost:30000/v1"
MODEL_NAME = "default"
MCP_SERVER_SCRIPT = "mcp_server_standalone.py"

# -----------------------------------------------------------------------------
# 2. 도구 정의 (Qwen에게 보여줄 메뉴판)
# -----------------------------------------------------------------------------
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_modis_images",
            "description": "[CRITICAL STEP 1] Retrieve a list of already generated/cached MODIS images. ALWAYS call this FIRST.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_modis_image",
            "description": "[STEP 2-A] Retrieve image data for a specific filename found in list.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_modis_data",
            "description": "[STEP 2-B] Fetch NEW MODIS imagery. WARNING: SLOW. Use only if image not in list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_str": {"type": "string", "description": "YYYY-MM-DD"},
                    "satellite": {"type": "string", "enum": ["terra", "aqua"]},
                    "west": {"type": "number"}, "south": {"type": "number"},
                    "east": {"type": "number"}, "north": {"type": "number"},
                    "is_daytime": {"type": "boolean"},
                    "pinpoint_lat": {"type": "number"}, "pinpoint_lng": {"type": "number"}
                },
                "required": ["date_str", "satellite", "west", "south", "east", "north"]
            }
        }
    }
]

# -----------------------------------------------------------------------------
# 3. 핵심 분석 프롬프트 (이미지 주입 시점에 사용)
# -----------------------------------------------------------------------------
FINAL_ANALYSIS_INSTRUCTION = """
## Role & Task
You are an expert meteorologist. The requested satellite imagery has been retrieved.
Your task is to synthesize a comprehensive analysis based on this visual data.

## Instructions
1. **Analyze** the provided MODIS satellite image (RGB + LST with purple dot).
2. **Think step-by-step** about the observation - WRAP ALL YOUR REASONING in <think></think> tags.
3. **Output Format**:
   - First, write your detailed analysis process inside <think>...</think> tags
   - Then, write your final answer OUTSIDE the tags
4. **Final Answer:** Provide a unified, professional weather report (6-10 sentences).
5. **Focus** on the most important findings and their implications.

"""

# -----------------------------------------------------------------------------
# 4. Mock Classes for Manual Parsing (SGLang 호환성 패치용)
# -----------------------------------------------------------------------------
class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments

class MockToolCall:
    def __init__(self, name, arguments):
        self.id = f"call_{name}_{os.urandom(4).hex()}"
        self.type = "function"
        self.function = MockFunction(name, arguments)

# -----------------------------------------------------------------------------
# 5. 유틸리티 함수
# -----------------------------------------------------------------------------
def determine_satellite_and_daytime(hour: int) -> tuple:
    """시간(hour)을 기반으로 위성(terra/aqua)과 주야간(daytime) 결정
    
    Args:
        hour: 0-23 범위의 시간
        
    Returns:
        (satellite, is_daytime) 튜플
    """
    if hour in [10, 11]:
        return 'terra', True
    elif hour in [13, 14]:
        return 'aqua', True
    elif hour in [22, 23]:
        return 'terra', False
    elif hour in [1, 2]:
        return 'aqua', False
    else:
        # 기본값
        return 'terra', True

# -----------------------------------------------------------------------------
# 6. 메인 VLM Agent 함수
# -----------------------------------------------------------------------------
async def run_vlm_agent_loop(user_query: str, system_prompt: str, original_question: str = None) -> tuple:
    """MCP Agent를 사용한 VLM 분석 루프
    
    Args:
        user_query: 사용자 질문 (location, date 정보 포함)
        system_prompt: 시스템 프롬프트
        original_question: 원본 사용자 질문 (이미지 분석 시 다시 상기시킴)
        
    Returns:
        (final_response, image_filename) 튜플
    """
    client = AsyncOpenAI(base_url=SGLANG_API_URL, api_key="EMPTY")

    # MCP 서버 프로세스 설정
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
        env=os.environ.copy()
    )

    logger.info(f"🔌 Connecting to MCP Server: {MCP_SERVER_SCRIPT}...")
    
    final_response = ""
    image_filename = None
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            logger.info(f"💬 User Query: {user_query}")

            for turn in range(6):  # 최대 6턴 (list -> fetch/get -> image analysis -> final response)
                logger.info(f"--- Turn {turn + 1} (Thinking...) ---")
                
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.4
                )
                
                msg = response.choices[0].message
                content = msg.content or ""

                # SGLang tool_calls 누락 시 수동 파싱
                if not msg.tool_calls:
                    if "<tool_call>" in content:
                        logger.warning("⚠️  Detected raw <tool_call> in text. Parsing manually...")
                        try:
                            pattern = r"<tool_call>(.*?)</tool_call>"
                            matches = re.findall(pattern, content, re.DOTALL)
                            
                            if matches:
                                msg.tool_calls = []
                                for match in matches:
                                    tool_json = json.loads(match.strip())
                                    if "name" in tool_json and "arguments" in tool_json:
                                        msg.tool_calls.append(
                                            MockToolCall(tool_json["name"], tool_json["arguments"])
                                        )
                        except Exception as e:
                            logger.error(f"❌ Manual parsing failed: {e}")

                # 도구 호출이 없으면 최종 답변
                if not msg.tool_calls:
                    logger.info(f"🤖 Final Answer received: {len(msg.content)} chars")
                    final_response = msg.content
                    break

                # 도구 호출 처리
                if isinstance(msg.tool_calls[0], MockToolCall):
                    tool_calls_dict = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in msg.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls_dict
                    })
                else:
                    messages.append(msg)
                
                # 도구 실행
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"🛠️  Model calls: {fn_name}")
                    logger.info(f"    Args: {fn_args}")

                    # MCP 서버에 실행 요청
                    result = await session.call_tool(fn_name, arguments=fn_args)
                    
                    # 결과 파싱
                    output_text = ""
                    output_data = {}
                    
                    for content_item in result.content:
                        if hasattr(content_item, "text"):
                            output_text += content_item.text
                            try:
                                output_data = json.loads(content_item.text)
                            except:
                                pass

                    # 이미지가 발견되면 분석 프롬프트 주입
                    if output_data.get("success") and "data_uri" in output_data:
                        logger.info("🖼️  Image retrieved! Injecting image & analysis prompt...")
                        
                        # 이미지 파일명 저장
                        image_filename = output_data.get("filename")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Image fetched successfully."
                        })

                        # 원본 질문을 다시 상기시킴
                        question_reminder = f"\n\nREMEMBER: The user's original question was: {original_question}" if original_question else ""
                        
                        image_msg = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": f"Here is the satellite image.\n\n{FINAL_ANALYSIS_INSTRUCTION}{question_reminder}"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": output_data["data_uri"]}
                                }
                            ]
                        }
                        messages.append(image_msg)
                        break
                    else:
                        logger.info(f"✅ Result: {output_text[:100]}...")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output_text
                        })
    
    return final_response, image_filename


async def analyze_weather_with_agent(
    query: str,
    bounds: Optional[dict] = None,
    datetime_str: Optional[str] = None,
    pinpoint: Optional[dict] = None
) -> tuple:
    """VLM Agent를 사용한 기상 분석
    
    Args:
        query: 사용자 질문
        bounds: 경계 좌표 {'west', 'south', 'east', 'north'}
        datetime_str: 날짜시간 문자열 (YYMMDDHH)
        pinpoint: 핀포인트 좌표 {'lat', 'lng'}
        
    Returns:
        (response_text, thinking_text, confidence, image_filename) 튜플
    """
    if not bounds or not datetime_str:
        return None, None, 0.85, None
    
    try:
        # datetime 파싱
        year = "20" + datetime_str[0:2]
        month = datetime_str[2:4]
        day = datetime_str[4:6]
        hour = int(datetime_str[6:8])
        
        # Rule-based satellite & daytime 결정
        satellite, is_daytime = determine_satellite_and_daytime(hour)
        
        # 날짜 문자열 생성
        date_str = f"{year}-{month}-{day}"
        
        # 쿼리 구성
        location_str = f"(west:{bounds['west']}, south:{bounds['south']}, east:{bounds['east']}, north:{bounds['north']})"
        if pinpoint:
            location_str += f" with pinpoint at (lat:{pinpoint['lat']}, lng:{pinpoint['lng']})"
        
        time_of_day = "daytime" if is_daytime else "nighttime"
        
        agent_query = f"""
Analyze the MODIS {satellite} {time_of_day} satellite imagery for the region {location_str} on {date_str}.

User's specific question: {query}
"""
        
        system_prompt = """You are a helpful meteorologist AI. Use tools to fetch satellite imagery when needed. 
Always check existing images first.
IMPORTANT: When calling fetch_modis_data, if a pinpoint location is mentioned in the query, you MUST include pinpoint_lat and pinpoint_lng parameters."""
        
        logger.info(f"🚀 Starting VLM Agent Loop...")
        logger.info(f"   Satellite: {satellite}, Daytime: {is_daytime}")
        logger.info(f"   Date: {date_str}, Location: {location_str}")
        
        # VLM Agent 실행 - 이미지 파일명도 반환 (원본 질문도 전달)
        agent_response, image_filename = await run_vlm_agent_loop(agent_query, system_prompt, original_question=query)
        
        # <think> 태그 파싱
        think_pattern = r"(.*?)</think>"
        think_matches = re.findall(think_pattern, agent_response, re.DOTALL)
        
        thinking_text = None
        response_text = agent_response
        
        if think_matches:
            thinking_text = think_matches[0].strip()
            # thinking 태그 제거하고 최종 응답만 추출
            response_text = re.sub(think_pattern, "", agent_response, flags=re.DOTALL).strip()
        
        logger.info("✅ VLM Agent 분석 완료")
        
        return response_text, thinking_text, 0.85, image_filename
        
    except Exception as e:
        logger.error(f"VLM Agent 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0.85, None
