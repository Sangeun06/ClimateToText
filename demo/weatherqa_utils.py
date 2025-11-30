"""
WeatherQA 데이터셋 유틸리티

ClimateToText 데이터셋의 WeatherQA 이미지를 찾고 관리합니다.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


# WeatherQA 데이터 경로
WEATHERQA_PATHS = [
    "/home/agi592/kse/ClimateToText/data/WeatherQA/WeatherQA_MD_2014-2019/md_image",
    "/home/agi592/kse/ClimateToText/data/WeatherQA/WeatherQA_MD_2020/md_image"
]

# WeatherQA 이미지 타입 정의 (실제 디렉토리명)
WEATHERQA_IMAGE_TYPES = [
    "bigsfc",
    "effh",
    "epvl",
    "fzlv",
    "laps",
    "lclh",
    "lllr",
    "mcon",
    "mcsm",
    "pchg",
    "rgnlrad",
    "sbcp",
    "scp",
    "shr6",
    "srh1",
    "stor",
    "swbt",
    "tadv"
]


def check_weatherqa_availability(date_str: str, hour: int) -> Dict:
    """
    특정 날짜/시간에 WeatherQA 이미지가 있는지 확인합니다.
    
    Args:
        date_str: 날짜 (YYYY-MM-DD)
        hour: 시간 (0-23)
    
    Returns:
        {
            'available': bool,
            'image_count': int,
            'image_types': List[str],
            'date_path': str
        }
    """
    try:
        # 날짜 파싱
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        
        # 파일명 패턴: md{id}_{YYYYMMDD}_{HH}_{type}.gif
        date_pattern = f"{year:04d}{month:02d}{day:02d}_{hour:02d}"
        
        # 적절한 경로 선택
        if year <= 2019:
            base_path = WEATHERQA_PATHS[0]
        else:
            base_path = WEATHERQA_PATHS[1]
        
        year_path = Path(base_path) / str(year)
        
        if not year_path.exists():
            return {
                'available': False,
                'image_count': 0,
                'image_types': [],
                'date_path': None
            }
        
        # 해당 시간대 이미지 찾기 (각 타입별 디렉토리에서 검색)
        found_types = []
        for img_type in WEATHERQA_IMAGE_TYPES:
            type_dir = year_path / img_type
            if not type_dir.exists():
                continue
            
            # 해당 날짜/시간의 파일이 있는지 확인
            # 파일명 패턴: md*_{YYYYMMDD}_{HH}_{type}.gif
            pattern = f"md*_{date_pattern}_{img_type}.gif"
            matching_files = list(type_dir.glob(pattern))
            
            if matching_files:
                found_types.append(img_type)
        
        return {
            'available': len(found_types) > 0,
            'image_count': len(found_types),
            'image_types': found_types,
            'date_path': str(year_path) if len(found_types) > 0 else None
        }
        
    except Exception as e:
        print(f"WeatherQA 가용성 확인 중 오류: {e}")
        return {
            'available': False,
            'image_count': 0,
            'image_types': [],
            'date_path': None
        }


def get_weatherqa_images(date_str: str, hour: int) -> List[Dict]:
    """
    특정 날짜/시간의 WeatherQA 이미지 정보를 반환합니다.
    
    Args:
        date_str: 날짜 (YYYY-MM-DD)
        hour: 시간 (0-23)
    
    Returns:
        List[{
            'type': str,
            'path': str,
            'url': str,  # 웹 접근용 URL
            'filename': str
        }]
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        
        date_pattern = f"{year:04d}{month:02d}{day:02d}_{hour:02d}"
        
        # 적절한 경로 선택
        if year <= 2019:
            base_path = WEATHERQA_PATHS[0]
        else:
            base_path = WEATHERQA_PATHS[1]
        
        year_path = Path(base_path) / str(year)
        
        if not year_path.exists():
            return []
        
        images = []
        for img_type in WEATHERQA_IMAGE_TYPES:
            type_dir = year_path / img_type
            if not type_dir.exists():
                continue
            
            # 해당 날짜/시간의 파일 찾기
            pattern = f"md*_{date_pattern}_{img_type}.gif"
            matching_files = list(type_dir.glob(pattern))
            
            if matching_files:
                # 첫 번째 매칭 파일 사용
                filepath = matching_files[0]
                filename = filepath.name
                
                # 웹 접근용 상대 경로 생성
                # Static mount 기준: /weatherqa/{year_range}/md_image/{year}/{type}/{file}
                if year <= 2019:
                    year_range = "WeatherQA_MD_2014-2019"
                else:
                    year_range = "WeatherQA_MD_2020"
                
                relative_path = f"{year_range}/md_image/{year}/{img_type}/{filename}"
                
                images.append({
                    'type': img_type,
                    'path': str(filepath),
                    'filename': filename,
                    'url': f"/weatherqa/{relative_path}"
                })
        
        return images
        
    except Exception as e:
        print(f"WeatherQA 이미지 로드 중 오류: {e}")
        return []


def get_image_type_description(img_type: str) -> str:
    """이미지 타입의 설명을 반환합니다."""
    descriptions = {
        "bigsfc": "Big Surface - 지표면 종합 분석",
        "effh": "Effective Helicity - 유효 나선도",
        "epvl": "Equivalent Potential Vorticity - 상당 잠재 와도",
        "fzlv": "Freezing Level - 어는점 고도",
        "laps": "Lapse Rate - 기온 감률",
        "lclh": "LCL Height - 상승 응결 고도",
        "lllr": "Low-Level Lapse Rate - 하층 기온 감률",
        "mcon": "Moisture Convergence - 수증기 수렴",
        "mcsm": "MCS Maintenance - 중규모 대류계 유지",
        "pchg": "Pressure Change - 기압 변화",
        "rgnlrad": "Regional Radar - 지역 레이더",
        "sbcp": "Surface-Based CAPE - 지표 기반 CAPE",
        "scp": "Supercell Composite Parameter - 슈퍼셀 종합 지수",
        "shr6": "0-6km Shear - 0-6km 연직 시어",
        "srh1": "Storm-Relative Helicity (0-1km) - 폭풍 상대 나선도",
        "stor": "Storm Motion - 폭풍 이동",
        "swbt": "Showalter Index - 쇼왈터 지수",
        "tadv": "Temperature Advection - 온도 이류"
    }
    return descriptions.get(img_type, img_type)
