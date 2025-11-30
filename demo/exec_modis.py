"""
MODIS VLM 실행 모듈

MODIS 이미지를 VLM에 전달하여 기상 분석을 수행합니다.
MCP 서버를 통해 VLM이 직접 이미지에 접근합니다.
"""

import logging
import requests
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# VLM API 설정 (OpenAI 호환 base URL)
VLM_API_URL = "http://localhost:30000/v1"
CHAT_COMPLETIONS_URL = f"{VLM_API_URL}/chat/completions"
VLM_MODEL_NAME = "output/qwen3_weather_merged"
MCP_SERVER_URL = "http://localhost:8001"  # MCP 서버 주소

SYSTEM_PROMPT = """You are an expert meteorologist and a senior data scientist. Your critical task is to answer the given question based on the provided satellite imagery.

You will be given:
1. A combined satellite image (RGB on the left, LST on the right, with a purple dot indicating the observation location).
2. A Question about the imagery.

## Instructions for Your Response
* Analyze the provided satellite imagery carefully.
* Think step-by-step about the observation, but keep your thinking process CONCISE (under 1500 tokens).
* Your final answer must be 4-6 sentences long, professional, confident, and directly address the question."""


def execute_modis_vlm(
    image_path: str,
    query: str,
    timeout: int = 120
) -> Tuple[Optional[str], Optional[str], float]:
    """
    MODIS 이미지를 VLM에 전달하여 분석합니다.
    이미지를 Base64로 인코딩하여 직접 전달합니다.
    
    Args:
        image_path: MODIS 이미지 파일 경로
        query: 사용자 질문
        timeout: API 타임아웃 (초)
    
    Returns:
        (response_text, thinking_text, confidence) 튜플
        실패 시 (error_message, None, 0.85)
    """
    
    try:
        # 이미지를 150x150으로 리사이즈한 후 Base64로 인코딩
        import base64
        from PIL import Image
        import io
        
        # 이미지 로드 및 리사이즈
        img = Image.open(image_path)
        img_resized = img.resize((300, 150), Image.Resampling.LANCZOS)
        
        # 리사이즈된 이미지를 바이트로 변환
        img_bytes = io.BytesIO()
        img_resized.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        image_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        
        # Data URI 형식으로 이미지 URL 생성
        image_data_uri = f"data:image/png;base64,{image_base64}"
        
        logger.info(f"VLM API 호출 중: {image_path} (Base64 인코딩 완료)")
        
        # API 요청 구성
        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_uri
                            }
                        },
                        {
                            "type": "text",
                            "text": f"## Your Task\n\nBased on the provided satellite imagery, answer the following question:\n\n{query}"
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2500
        }
        
        # API 호출
        response = requests.post(
            CHAT_COMPLETIONS_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            full_response = result['choices'][0]['message']['content']
            
            logger.info(f"VLM 응답 받음: {len(full_response)} 글자")
            
            # </think> 태그를 기준으로 사고 과정과 답변 분리
            thinking_text = None
            if '</think>' in full_response:
                parts = full_response.split('</think>')
                
                thinking_part = parts[0]
                if '<think>' in thinking_part:
                    thinking_text = thinking_part.replace('<think>', '').strip()
                else:
                    thinking_text = thinking_part.strip()
                
                response_text = parts[-1].strip()
                
                logger.info(f"사고 과정 추출: {len(thinking_text)} 글자, 답변: {len(response_text)} 글자")
            else:
                response_text = full_response.strip()
                logger.info("</think> 태그 없음 - 전체를 답변으로 사용")
            
            confidence = 0.92
            
            return response_text, thinking_text, confidence
        else:
            logger.error(f"VLM API 오류: {response.status_code} - {response.text}")
            return f"VLM 모델 응답 오류가 발생했습니다. (상태 코드: {response.status_code})", None, 0.85
            
    except requests.exceptions.Timeout:
        logger.error("VLM API 타임아웃")
        return "분석 시간이 초과되었습니다. 다시 시도해주세요.", None, 0.85
    except Exception as e:
        logger.error(f"VLM API 호출 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return f"모델 분석 중 오류가 발생했습니다: {str(e)}", None, 0.85


def summarize_weather_analyses(
    weatherqa_analyses: list,
    modis_analysis: Optional[str] = None,
    query: str = "",
    timeout: int = 120
) -> Tuple[Optional[str], Optional[str], float]:
    """
    WeatherQA 분석 결과들과 MODIS 분석을 종합하여 최종 요약을 생성합니다.
    
    Args:
        weatherqa_analyses: WeatherQA 분석 결과 리스트 [{'type': '...', 'description': '...', 'generated': '...'}, ...]
        modis_analysis: MODIS VLM 분석 결과 (선택)
        query: 사용자 질문
        timeout: API 타임아웃 (초)
    
    Returns:
        (summary_text, thinking_text, confidence) 튜플
        실패 시 (error_message, None, 0.85)
    """
    
    try:
        # 요약을 위한 시스템 프롬프트
        summary_prompt = """You are an expert meteorologist. Your task is to synthesize multiple meteorological analyses into a comprehensive, coherent summary.

You will be given:
1. Multiple analyses of different meteorological parameters (WeatherQA)
2. Optionally, a satellite image analysis (MODIS)
3. A user question

## Instructions for Your Response
* Think step-by-step about how to synthesize the analyses, but keep your thinking process CONCISE (under 1500 tokens)
* Wrap your thinking process in <think></think> tags
* After your thinking, provide a unified, professional weather report (6-10 sentences)
* Focus on the most important findings and their implications
* Do NOT simply list the individual analyses - create a cohesive narrative that directly addresses the user's question"""

        # 입력 텍스트 구성
        input_text = f"## User Question\n{query}\n\n"
        
        # WeatherQA 분석 추가
        if weatherqa_analyses:
            input_text += "## Meteorological Parameter Analyses (WeatherQA)\n\n"
            for item in weatherqa_analyses:
                img_type = item.get('type', 'Unknown')
                desc = item.get('description', '')
                generated = item.get('generated', '')
                
                input_text += f"**{img_type}** ({desc}):\n{generated}\n\n"
        
        # MODIS 분석 추가
        if modis_analysis:
            input_text += f"## Satellite Imagery Analysis (MODIS)\n\n{modis_analysis}\n\n"
        
        input_text += "## Your Task\n\nBased on all the above analyses, provide a comprehensive weather summary that directly answers the user's question."
        
        logger.info(f"종합 요약 VLM API 호출 중 (입력: {len(input_text)} 글자)")
        
        # API 요청 구성
        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": summary_prompt
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        # API 호출
        response = requests.post(
            VLM_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            full_response = result['choices'][0]['message']['content'].strip()
            
            logger.info(f"종합 요약 응답 받음: {len(full_response)} 글자")
            
            # </think> 태그를 기준으로 사고 과정과 답변 분리
            thinking_text = None
            if '</think>' in full_response:
                parts = full_response.split('</think>')
                
                thinking_part = parts[0]
                if '<think>' in thinking_part:
                    thinking_text = thinking_part.replace('<think>', '').strip()
                else:
                    thinking_text = thinking_part.strip()
                
                summary_text = parts[-1].strip()
                
                logger.info(f"사고 과정 추출: {len(thinking_text)} 글자, 요약: {len(summary_text)} 글자")
            else:
                summary_text = full_response
                logger.info("</think> 태그 없음 - 전체를 요약으로 사용")
            
            confidence = 0.95
            return summary_text, thinking_text, confidence
        else:
            logger.error(f"종합 요약 VLM API 오류: {response.status_code} - {response.text}")
            return f"종합 요약 생성 중 오류가 발생했습니다. (상태 코드: {response.status_code})", None, 0.85
            
    except requests.exceptions.Timeout:
        logger.error("종합 요약 VLM API 타임아웃")
        return "종합 요약 시간이 초과되었습니다. 다시 시도해주세요.", None, 0.85
    except Exception as e:
        logger.error(f"종합 요약 VLM API 호출 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return f"종합 요약 중 오류가 발생했습니다: {str(e)}", None, 0.85
