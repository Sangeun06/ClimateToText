from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from datetime import datetime
from typing import Optional
import random
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
import json
import re
import base64
import os

# MODIS 함수 가져오기 (Earth Engine 초기화, 데이터 가용성 확인만 직접 사용)
from get_modis import initialize_earth_engine, check_data_availability
from weatherqa_utils import get_weatherqa_images, get_image_type_description
from weatherqa_inference import infer_weatherqa_images
from exec_modis import summarize_weather_analyses
# exec_modis는 더 이상 직접 import하지 않음 - MCP 서버가 처리

app = FastAPI(title="기상 AI 서비스")
# Run vlm: python -m sglang.launch_server --model-path EarthData/output/qwen3_weather_merged --port 30000 --host 0.0.0.0
# Run mcp: python mcp_server.py

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Earth Engine 초기화 (서버 시작 시)
@app.on_event("startup")
async def startup_event():
    logger.info("Earth Engine 초기화 중...")
    if initialize_earth_engine():
        logger.info("Earth Engine 초기화 성공")
    else:
        logger.warning("Earth Engine 초기화 실패 - MODIS 기능이 제한될 수 있습니다.")

# --- 여기부터 추가/수정하세요 ---
origins = [
    "http://localhost:8000", # 프론트엔드가 실행되는 정확한 주소
    "http://127.0.0.1:8000",
    # 필요하다면 "*" 로 모든 접근을 허용할 수도 있지만, 보안상 특정하는 것이 좋습니다.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 허용할 출처 목록
    allow_credentials=True,     # 쿠키/인증 정보 포함 여부
    allow_methods=["*"],        # 허용할 HTTP 메소드 (GET, POST 등)
    allow_headers=["*"],        # 허용할 헤더
)
# ---------------------------

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 간단한 AI 응답 생성 함수 (실제로는 모델을 사용)
def generate_weather_response(
    query: str, 
    bounds: Optional[dict] = None, 
    datetime_str: Optional[str] = None,
    source: Optional[str] = None,
    pinpoint: Optional[dict] = None,
    image_path: Optional[str] = None
) -> dict:
    """VLM API를 사용한 기상 분석 응답 생성"""
    
    # 날짜 포맷팅
    datetime_formatted = "시간 정보 없음"
    if datetime_str and len(datetime_str) == 8:
        try:
            year = "20" + datetime_str[0:2]
            month = datetime_str[2:4]
            day = datetime_str[4:6]
            hour = datetime_str[6:8]
            datetime_formatted = f"{year}년 {month}월 {day}일 {hour}시"
        except:
            pass
    
    # 영역 포맷팅 (XYXY)
    bounds_xyxy = "영역 정보 없음"
    if bounds:
        bounds_xyxy = f"X: [{bounds.get('west')}, {bounds.get('east')}], Y: [{bounds.get('south')}, {bounds.get('north')}]"
    
    # 핀포인트 포맷팅
    pinpoint_formatted = None
    if pinpoint:
        pinpoint_formatted = f"위도: {pinpoint.get('lat')}°, 경도: {pinpoint.get('lng')}°"
    
    # 데이터 소스
    source_name = source.upper() if source else "알 수 없음"
    
    # VLM API 호출 (이미지가 있을 때만) - MCP 서버를 통해 처리
    response_text = None
    thinking_text = None
    confidence = 0.85
    
    if image_path and source and 'modis' in source.lower():
        # MCP 서버의 analyze_satellite_image 도구 호출
        try:
            analyze_response = requests.post(
                "http://localhost:8001/call_tool",
                json={
                    "tool_name": "analyze_satellite_image",
                    "arguments": {
                        "image_path": image_path,
                        "query": query
                    }
                },
                timeout=180
            )
            
            if analyze_response.status_code == 200:
                analyze_result = analyze_response.json()
                if analyze_result.get('success'):
                    response_text = analyze_result.get('response')
                    thinking_text = analyze_result.get('thinking')
                    confidence = analyze_result.get('confidence', 0.85)
                    logger.info("MCP 서버를 통한 VLM 분석 완료")
                else:
                    logger.warning(f"MCP VLM 분석 실패: {analyze_result.get('error')}")
                    response_text = analyze_result.get('response', "분석 중 오류가 발생했습니다.")
            else:
                logger.error(f"MCP 서버 분석 호출 실패: {analyze_response.status_code}")
                
        except Exception as e:
            logger.error(f"MCP 서버 분석 호출 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # VLM 응답이 없으면 기본 응답
    if not response_text:
        pinpoint_info = f" 선택하신 지점({pinpoint_formatted})" if pinpoint else ""
        response_text = f"{source_name} 데이터를 사용하여{pinpoint_info}의 {datetime_formatted} 기상 정보를 분석했습니다."
    
    result = {
        "query": query,
        "response": response_text,
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bounds": bounds,
        "bounds_xyxy": bounds_xyxy,
        "datetime": datetime_str,
        "datetime_formatted": datetime_formatted,
        "source": source_name,
        "pinpoint": pinpoint,
        "pinpoint_formatted": pinpoint_formatted
    }
    
    # 사고 과정이 있으면 추가
    if thinking_text:
        result['thinking'] = thinking_text
    
    return result


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/check_availability")
async def check_availability(datetime: str):
    """데이터 소스 사용 가능 여부 확인 API"""
    try:
        # datetime 파싱 (YYMMDDHH 형식)
        if len(datetime) != 8:
            return {
                "datetime": datetime,
                "modis_available": False,
                "cli2text_available": True
            }
        
        year = int("20" + datetime[0:2])
        month = int(datetime[2:4])
        day = int(datetime[4:6])
        hour = int(datetime[6:8])
        
        # 날짜 문자열 생성
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        
        # 미국 본토 영역 (Cli2Text 기본 영역)
        bbox = (-125.0, 24.0, -66.0, 49.0)
        
        # MODIS 데이터 가용성 확인
        availability = check_data_availability(date_str, hour, bbox)
        
        return {
            "datetime": datetime,
            "modis_available": availability['modis_available'],
            "cli2text_available": availability['cli2text_available'],
            "satellite": availability.get('satellite'),
            "weatherqa_image_count": availability.get('weatherqa_image_count', 0),
            "weatherqa_types": availability.get('weatherqa_types', [])
        }
        
    except Exception as e:
        logger.error(f"데이터 가용성 확인 중 오류: {e}")
        return {
            "datetime": datetime,
            "modis_available": False,
            "cli2text_available": True
        }


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request, 
    query: str = Form(...),
    bounds: Optional[str] = Form(None),
    datetime: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    pinpoint: Optional[str] = Form(None)
):
    """기상 AI 예측 엔드포인트"""
    import json
    
    # JSON 문자열을 dict로 변환
    bounds_dict = None
    if bounds:
        try:
            bounds_dict = json.loads(bounds)
        except:
            pass
    
    pinpoint_dict = None
    if pinpoint:
        try:
            pinpoint_dict = json.loads(pinpoint)
        except:
            pass
    
    # MODIS 소스가 선택되었으면 실제 이미지 가져오기
    image_path = None
    weatherqa_images = []
    if source and datetime and bounds_dict:
        try:
            # datetime 파싱 (YYMMDDHH)
            year = int("20" + datetime[0:2])
            month = int(datetime[2:4])
            day = int(datetime[4:6])
            hour = int(datetime[6:8])
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            
            # MODIS 처리
            if 'modis' in source.lower():
                # 위성 및 주/야간 결정
                is_daytime = hour in [10, 11, 13, 14]
                if hour in [10, 11]:
                    satellite = 'terra'
                elif hour in [13, 14]:
                    satellite = 'aqua'
                elif hour in [22, 23]:
                    satellite = 'terra'
                    is_daytime = False
                elif hour in [1, 2]:
                    satellite = 'aqua'
                    is_daytime = False
                else:
                    satellite = 'terra'
                
                # bbox 추출
                bbox = (
                    float(bounds_dict['west']),
                    float(bounds_dict['south']),
                    float(bounds_dict['east']),
                    float(bounds_dict['north'])
                )
                
                # pinpoint 추출
                pinpoint_tuple = None
                if pinpoint_dict:
                    pinpoint_tuple = (
                        float(pinpoint_dict['lat']),
                        float(pinpoint_dict['lng'])
                    )
                
                logger.info(f"MODIS 이미지 요청: {date_str}, {satellite}, {bbox}, pinpoint={pinpoint_tuple}")
                
                # MCP 서버를 통해 MODIS 이미지 가져오기
                import requests
                mcp_response = requests.post(
                    "http://localhost:8001/call_tool",
                    json={
                        "tool_name": "fetch_modis_data",
                        "arguments": {
                            "date_str": date_str,
                            "satellite": satellite,
                            "west": bbox[0],
                            "south": bbox[1],
                            "east": bbox[2],
                            "north": bbox[3],
                            "is_daytime": is_daytime,
                            "pinpoint_lat": pinpoint_tuple[0] if pinpoint_tuple else None,
                            "pinpoint_lng": pinpoint_tuple[1] if pinpoint_tuple else None
                        }
                    },
                    timeout=180
                )
                
                if mcp_response.status_code == 200:
                    mcp_result = mcp_response.json()
                    if mcp_result.get('success'):
                        # MCP 서버가 반환한 파일명으로 경로 구성
                        filename = mcp_result.get('filename')
                        image_path = os.path.join('./assets', filename)
                        logger.info(f"MCP 서버를 통해 이미지 생성 성공: {image_path}")
                    else:
                        logger.warning(f"MCP 서버 이미지 생성 실패: {mcp_result.get('error')}")
                else:
                    logger.error(f"MCP 서버 호출 실패: {mcp_response.status_code}")
            
            # Cli2Text/WeatherQA 처리
            if 'cli2text' in source.lower():
                logger.info(f"WeatherQA 이미지 로드: {date_str}, {hour}시")
                weatherqa_raw = get_weatherqa_images(date_str, hour)
                
                # WeatherQA 추론 수행
                logger.info(f"WeatherQA 추론 시작 ({len(weatherqa_raw)}개 이미지)")
                weatherqa_images = infer_weatherqa_images(weatherqa_raw)
                
                # 각 이미지에 설명 추가
                for img in weatherqa_images:
                    img['description'] = get_image_type_description(img['type'])
                
                logger.info(f"WeatherQA 추론 완료: {len(weatherqa_images)}개")

                
        except Exception as e:
            logger.error(f"MODIS 이미지 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    result = generate_weather_response(
        query=query, 
        bounds=bounds_dict, 
        datetime_str=datetime, 
        source=source, 
        pinpoint=pinpoint_dict,
        image_path=image_path
    )
    
    # 이미지 경로가 있으면 결과에 추가
    if image_path:
        result['image_path'] = image_path
    
    # WeatherQA 이미지가 있으면 결과에 추가
    if weatherqa_images:
        result['weatherqa_images'] = weatherqa_images
        
        # WeatherQA가 있는 경우 종합 요약 생성
        logger.info("WeatherQA 종합 요약 생성 중...")
        modis_response = result.get('response') if image_path else None
        
        # MODIS 응답을 별도로 저장
        if modis_response:
            result['modis_response'] = modis_response
        
        summary_text, summary_thinking, summary_confidence = summarize_weather_analyses(
            weatherqa_analyses=weatherqa_images,
            modis_analysis=modis_response,
            query=query,
            timeout=120
        )
        
        if summary_text:
            # 종합 요약으로 메인 응답 교체
            result['response'] = summary_text
            result['confidence'] = summary_confidence
            result['has_summary'] = True
            
            # 종합 요약의 사고 과정 추가
            if summary_thinking:
                result['summary_thinking'] = summary_thinking
            
            logger.info(f"종합 요약 생성 완료: {len(summary_text)} 글자")
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "result": result
        }
    )


@app.get("/api/predict")
async def api_predict(
    query: str, 
    bounds: Optional[str] = None, 
    datetime: Optional[str] = None,
    source: Optional[str] = None,
    pinpoint: Optional[str] = None
):
    """API 엔드포인트 (JSON 응답)"""
    import json
    
    bounds_dict = None
    if bounds:
        try:
            bounds_dict = json.loads(bounds)
        except:
            pass
    
    pinpoint_dict = None
    if pinpoint:
        try:
            pinpoint_dict = json.loads(pinpoint)
        except:
            pass
    
    return generate_weather_response(query, bounds_dict, datetime, source, pinpoint_dict)


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "service": "Weather AI Service"}


# Static files (for serving generated images) - 마운트는 라우트 정의 후에
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
# WeatherQA 이미지 서빙
app.mount("/weatherqa", StaticFiles(directory="/home/agi592/kse/ClimateToText/data/WeatherQA"), name="weatherqa")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
