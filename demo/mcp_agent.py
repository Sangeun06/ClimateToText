import asyncio
import json
import os
import sys
import re  # [필수] 정규표현식 모듈
from typing import List, Dict, Any

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# -----------------------------------------------------------------------------
# 1. 설정 (환경에 맞게 수정하세요)
# -----------------------------------------------------------------------------
SGLANG_API_URL = "http://localhost:30000/v1"
MODEL_NAME = "default"  # SGLang 로드 모델 (보통 default)
MCP_SERVER_SCRIPT = "mcp_server_standalone.py"  # 같은 폴더에 있어야 함

# -----------------------------------------------------------------------------
# 2. 도구 정의 (Qwen에게 보여줄 메뉴판 - Docstring과 일치시킴)
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
2. **Think step-by-step** about the observation.
3. **Wrap your thinking process** in <think></think> tags.
4. **Final Output:** Provide a unified, professional weather report (6-10 sentences).
5. **Focus** on the most important findings and their implications.
6. **Create a cohesive narrative** that directly addresses the user's original question.
"""

# -----------------------------------------------------------------------------
# 4. Mock Classes for Manual Parsing (SGLang 호환성 패치용)
# -----------------------------------------------------------------------------
class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        # 인자가 이미 dict라면 string으로 변환, string이면 그대로 사용
        self.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments

class MockToolCall:
    def __init__(self, name, arguments):
        self.id = f"call_{name}_{os.urandom(4).hex()}"
        self.type = "function"
        self.function = MockFunction(name, arguments)

# -----------------------------------------------------------------------------
# 5. 메인 파이프라인 (MCP Host Implementation)
# -----------------------------------------------------------------------------
async def run_vlm_agent_loop(user_query: str, system_prompt: str):
    client = AsyncOpenAI(base_url=SGLANG_API_URL, api_key="EMPTY")

    # MCP 서버 프로세스 설정
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
        env=os.environ.copy()
    )

    print(f"🔌 Connecting to MCP Server: {MCP_SERVER_SCRIPT}...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            print(f"\n💬 User Query: {user_query}")

            for turn in range(3):
                print(f"\n--- Turn {turn + 1} (Thinking...) ---")
                
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.1
                )
                
                msg = response.choices[0].message
                content = msg.content or ""


                if not msg.tool_calls:
                    # 1. <tool_call> 태그 확인
                    if "<tool_call>" in content:
                        print("⚠️  Detected raw <tool_call> in text. Parsing manually...")
                        try:
                            # 정규표현식으로 태그 안의 JSON 내용 추출
                            pattern = r"<tool_call>(.*?)</tool_call>"
                            matches = re.findall(pattern, content, re.DOTALL)
                            
                            if matches:
                                msg.tool_calls = []
                                for match in matches:
                                    # JSON 파싱
                                    tool_json = json.loads(match.strip())
                                    
                                    # name과 arguments가 있는지 확인
                                    if "name" in tool_json and "arguments" in tool_json:
                                        msg.tool_calls.append(
                                            MockToolCall(tool_json["name"], tool_json["arguments"])
                                        )
                                    else:
                                        print(f"❌ Invalid tool call format: {tool_json}")

                        except Exception as e:
                            print(f"❌ Manual parsing failed: {e}")
                            print(f"   Content was: {content}")

                # =================================================================

                # 여전히 도구 호출이 없으면 -> 진짜 답변으로 간주하고 종료
                if not msg.tool_calls:
                    print(f"🤖 Final Answer:\n{msg.content}")
                    return

                # 도구 호출 로직 진행
                if isinstance(msg.tool_calls[0], MockToolCall):
                    # Mock 객체인 경우 수동으로 dict 구성하여 대화 내역에 추가
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
                        "content": content, # 원본 텍스트(<tool_call> 포함) 유지
                        "tool_calls": tool_calls_dict
                    })
                else:
                    # 정상적인 OpenAI 객체라면 그대로 append
                    messages.append(msg)
                
                # 도구 실행
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🛠️  Model calls: {fn_name}")
                    print(f"    Args: {fn_args}")

                    # MCP 서버에 실행 요청
                    result = await session.call_tool(fn_name, arguments=fn_args)
                    
                    # 결과 파싱
                    output_text = ""
                    output_data = {}
                    
                    # MCP SDK 응답 구조 처리
                    for content_item in result.content:
                        if hasattr(content_item, "text"):
                            output_text += content_item.text
                            try:
                                output_data = json.loads(content_item.text)
                            except:
                                pass

                    # -------------------------------------------------------
                    # [핵심] 이미지가 발견되면 프롬프트 주입 및 태세 전환
                    # -------------------------------------------------------
                    if output_data.get("success") and "data_uri" in output_data:
                        print("🖼️  Image retrieved! Injecting image & specific analysis prompt...")
                        
                        # 1. Tool 결과 메시지 (성공 기록)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Image fetched successfully."
                        })

                        # 2. 이미지 + 전문가 분석 프롬프트 주입
                        image_msg = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": f"Here is the satellite image.\n\n{FINAL_ANALYSIS_INSTRUCTION}"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": output_data["data_uri"]}
                                }
                            ]
                        }
                        messages.append(image_msg)
                        
                        # 이미지를 찾았으므로 다음 턴(분석)으로 바로 넘김
                        break 
                    
                    else:
                        # 이미지가 없는 일반 응답
                        print(f"✅ Result: {output_text[:100]}...")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output_text
                        })

# -----------------------------------------------------------------------------
# 6. 실행
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 질문 예시
    query = """
    Analyze the MODIS terra daytime satellite imagery for California on August 15, 2023. (west:-124, south:32.5, east:-114, north:42)
    """

    
    # 초기 시스템 프롬프트는 가볍게 설정
    sys_prompt = "You are a helpful meteorologist AI. Use tools to fetch satellite imagery when needed. Always check existing images first."

    try:
        asyncio.run(run_vlm_agent_loop(query, sys_prompt))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")