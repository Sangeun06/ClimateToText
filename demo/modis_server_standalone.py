"""
MCP 서버 - Standalone FastAPI 버전

FastAPI를 사용하여 VLM이 호출할 수 있는 도구를 제공합니다.
VLM이 직접 MODIS 이미지를 생성할 수 있도록 get_modis 함수를 호출합니다.
"""

import os
import base64
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import Optional, Tuple
from datetime import datetime

# get_modis 모듈 임포트
import get_modis
# exec_modis 모듈 임포트 (VLM 분석 기능)
from exec_modis import execute_modis_vlm

app = FastAPI(title="MODIS MCP Server", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Assets 디렉토리 경로
ASSETS_DIR = Path(__file__).parent / "assets"

# 도구 호출 요청 모델
class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict

# 도구 정의
TOOLS = {
    "fetch_modis_data": {
        "description": "Fetches MODIS satellite data for a specific date, location, and satellite. Creates a combined RGB+LST image.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_str": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (e.g., '2023-04-14')"
                },
                "satellite": {
                    "type": "string",
                    "enum": ["terra", "aqua"],
                    "description": "Satellite name: 'terra' or 'aqua'"
                },
                "west": {
                    "type": "number",
                    "description": "Western longitude boundary"
                },
                "south": {
                    "type": "number",
                    "description": "Southern latitude boundary"
                },
                "east": {
                    "type": "number",
                    "description": "Eastern longitude boundary"
                },
                "north": {
                    "type": "number",
                    "description": "Northern latitude boundary"
                },
                "is_daytime": {
                    "type": "boolean",
                    "description": "True for daytime data, False for nighttime"
                },
                "pinpoint_lat": {
                    "type": "number",
                    "description": "Latitude of pinpoint marker (optional)"
                },
                "pinpoint_lng": {
                    "type": "number",
                    "description": "Longitude of pinpoint marker (optional)"
                }
            },
            "required": ["date_str", "satellite", "west", "south", "east", "north", "is_daytime"]
        }
    },
    "get_modis_image": {
        "description": "Retrieves a MODIS satellite image file in Base64 format",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The MODIS image filename"
                }
            },
            "required": ["filename"]
        }
    },
    "list_modis_images": {
        "description": "Lists all available MODIS satellite images",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "check_data_availability": {
        "description": "Checks if MODIS data is available for a specific date, time, and location",
        "parameters": {
            "type": "object",
            "properties": {
                "date_str": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format"
                },
                "hour": {
                    "type": "integer",
                    "description": "Hour of the day (0-23)"
                },
                "west": {
                    "type": "number",
                    "description": "Western longitude boundary"
                },
                "south": {
                    "type": "number",
                    "description": "Southern latitude boundary"
                },
                "east": {
                    "type": "number",
                    "description": "Eastern longitude boundary"
                },
                "north": {
                    "type": "number",
                    "description": "Northern latitude boundary"
                }
            },
            "required": ["date_str", "hour", "west", "south", "east", "north"]
        }
    },
    "analyze_satellite_image": {
        "description": "Analyzes a MODIS satellite image using VLM and returns weather analysis results",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the MODIS satellite image file"
                },
                "query": {
                    "type": "string",
                    "description": "Question or analysis request about the satellite imagery"
                }
            },
            "required": ["image_path", "query"]
        }
    }
}

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "service": "MCP Server"}

@app.get("/tools")
async def list_tools():
    """사용 가능한 도구 목록 반환"""
    return {"tools": TOOLS}

@app.post("/call_tool")
async def call_tool(request: ToolCallRequest):
    """도구 호출 엔드포인트"""
    tool_name = request.tool_name
    arguments = request.arguments
    
    if tool_name == "fetch_modis_data":
        return fetch_modis_data(
            date_str=arguments.get("date_str", ""),
            satellite=arguments.get("satellite", "terra"),
            west=arguments.get("west", 0.0),
            south=arguments.get("south", 0.0),
            east=arguments.get("east", 0.0),
            north=arguments.get("north", 0.0),
            is_daytime=arguments.get("is_daytime", True),
            pinpoint_lat=arguments.get("pinpoint_lat"),
            pinpoint_lng=arguments.get("pinpoint_lng")
        )
    elif tool_name == "get_modis_image":
        return get_modis_image(arguments.get("filename", ""))
    elif tool_name == "list_modis_images":
        return list_modis_images()
    elif tool_name == "check_data_availability":
        return check_data_availability(
            date_str=arguments.get("date_str", ""),
            hour=arguments.get("hour", 0),
            west=arguments.get("west", 0.0),
            south=arguments.get("south", 0.0),
            east=arguments.get("east", 0.0),
            north=arguments.get("north", 0.0)
        )
    elif tool_name == "analyze_satellite_image":
        return analyze_satellite_image(
            image_path=arguments.get("image_path", ""),
            query=arguments.get("query", "")
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}

def get_modis_image(filename: str) -> dict:
    """
    MODIS 이미지 파일을 Base64로 인코딩하여 반환합니다.
    
    Args:
        filename: 이미지 파일명
    
    Returns:
        {image_base64, mime_type, filename, data_uri}
    """
    try:
        image_path = ASSETS_DIR / filename
        
        if not image_path.exists():
            return {
                "error": f"Image file not found: {filename}",
                "available_files": [f.name for f in ASSETS_DIR.glob("*.png")]
            }
        
        # 이미지를 Base64로 인코딩
        with open(image_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Data URI 형식
        data_uri = f"data:image/png;base64,{image_base64}"
        
        return {
            "success": True,
            "filename": filename,
            "mime_type": "image/png",
            "image_base64": image_base64,
            "data_uri": data_uri,
            "size_kb": round(len(image_base64) / 1024, 2)
        }
        
    except Exception as e:
        return {
            "error": f"Failed to read image: {str(e)}",
            "filename": filename
        }

def list_modis_images() -> dict:
    """
    Assets 디렉토리의 모든 MODIS 이미지 목록을 반환합니다.
    
    Returns:
        {files: list, count: int}
    """
    try:
        if not ASSETS_DIR.exists():
            return {
                "error": "Assets directory not found",
                "path": str(ASSETS_DIR)
            }
        
        # PNG 파일만 필터링
        image_files = [f.name for f in ASSETS_DIR.glob("*.png")]
        
        return {
            "success": True,
            "files": sorted(image_files),
            "count": len(image_files),
            "directory": str(ASSETS_DIR)
        }
        
    except Exception as e:
        return {
            "error": f"Failed to list images: {str(e)}"
        }


def fetch_modis_data(
    date_str: str,
    satellite: str,
    west: float,
    south: float,
    east: float,
    north: float,
    is_daytime: bool = True,
    pinpoint_lat: Optional[float] = None,
    pinpoint_lng: Optional[float] = None
) -> dict:
    """
    MODIS 데이터를 가져와 RGB+LST 합성 이미지를 생성합니다.
    
    Args:
        date_str: 날짜 (YYYY-MM-DD)
        satellite: 'terra' 또는 'aqua'
        west, south, east, north: 바운딩 박스
        is_daytime: 주간/야간 구분
        pinpoint_lat, pinpoint_lng: 핀포인트 좌표 (선택)
    
    Returns:
        {success, filename, image_base64, data_uri} 또는 {error}
    """
    try:
        bbox = (west, south, east, north)
        pinpoint = (pinpoint_lat, pinpoint_lng) if pinpoint_lat and pinpoint_lng else None
        
        # get_modis.fetch_modis_images 호출
        result_path = get_modis.fetch_modis_images(
            date_str=date_str,
            satellite=satellite,
            bbox=bbox,
            is_daytime=is_daytime,
            output_dir=str(ASSETS_DIR),
            image_size=(512, 512),  # 고해상도로 가져오기
            pinpoint=pinpoint
        )
        
        if result_path:
            # 생성된 이미지를 Base64로 인코딩
            filename = os.path.basename(result_path)
            with open(result_path, 'rb') as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            data_uri = f"data:image/png;base64,{image_base64}"
            
            return {
                "success": True,
                "filename": filename,
                "image_base64": image_base64,
                "data_uri": data_uri,
                "size_kb": round(len(image_base64) / 1024, 2),
                "message": f"Successfully fetched MODIS data for {date_str} ({satellite})"
            }
        else:
            return {
                "error": "Failed to fetch MODIS data",
                "date": date_str,
                "satellite": satellite
            }
            
    except Exception as e:
        import traceback
        return {
            "error": f"Exception while fetching MODIS data: {str(e)}",
            "traceback": traceback.format_exc()
        }


def check_data_availability(
    date_str: str,
    hour: int,
    west: float,
    south: float,
    east: float,
    north: float
) -> dict:
    """
    MODIS 데이터 가용성을 확인합니다.
    
    Returns:
        {modis_available, cli2text_available, satellite}
    """
    try:
        bbox = (west, south, east, north)
        result = get_modis.check_data_availability(date_str, hour, bbox)
        return result
    except Exception as e:
        return {
            "error": f"Failed to check availability: {str(e)}",
            "modis_available": False,
            "cli2text_available": True
        }


def analyze_satellite_image(
    image_path: str,
    query: str
) -> dict:
    """
    저장된 위성 이미지를 VLM으로 분석하여 결과를 반환합니다.
    
    Args:
        image_path: MODIS 이미지 파일 경로
        query: 분석 질문
    
    Returns:
        {success, response, thinking, confidence} 또는 {error}
    """
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"VLM 분석 시작: {image_path}")
        
        # exec_modis 모듈의 VLM 실행 함수 호출
        response_text, thinking_text, confidence = execute_modis_vlm(
            image_path=image_path,
            query=query,
            timeout=120
        )
        
        if response_text:
            logger.info(f"VLM 분석 완료: {len(response_text)} 글자")
            return {
                "success": True,
                "response": response_text,
                "thinking": thinking_text,
                "confidence": confidence
            }
        else:
            return {
                "error": "VLM analysis failed",
                "response": "분석 중 오류가 발생했습니다.",
                "thinking": None,
                "confidence": 0.5
            }
            
    except Exception as e:
        import traceback
        return {
            "error": f"Exception during VLM analysis: {str(e)}",
            "traceback": traceback.format_exc(),
            "response": f"분석 중 오류가 발생했습니다: {str(e)}",
            "thinking": None,
            "confidence": 0.5
        }


if __name__ == "__main__":
    import uvicorn
    
    # Earth Engine 초기화
    print("🌍 Earth Engine 초기화 중...")
    get_modis.initialize_earth_engine()
    
    print("🚀 MCP 서버 시작 중...")
    print(f"📁 Assets 디렉토리: {ASSETS_DIR}")
    print("🛠️  사용 가능한 도구:")
    print("   - fetch_modis_data: MODIS 이미지 생성")
    print("   - get_modis_image: 저장된 이미지 로드")
    print("   - list_modis_images: 이미지 목록 조회")
    print("   - check_data_availability: 데이터 가용성 확인")
    print("   - analyze_satellite_image: VLM 기반 위성 이미지 분석 ⭐")
    uvicorn.run(app, host="0.0.0.0", port=8001)
