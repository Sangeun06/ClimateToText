"""
MODIS 데이터 가져오기 및 이미지 생성 모듈

특정 날짜, 위성(Terra/Aqua), XYXY 좌표가 주어졌을 때
RGB와 LST 이미지를 가져와 하나로 합쳐서 저장합니다.
"""

import ee
import requests
import io
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional
from datetime import datetime
from weatherqa_utils import check_weatherqa_availability


def initialize_earth_engine():
    """Google Earth Engine 초기화"""
    try:
        ee.Initialize()
        print("Google Earth Engine에 성공적으로 연결되었습니다.")
        return True
    except Exception as e:
        print(f"Earth Engine 초기화 실패: {e}")
        ee.Authenticate(auth_mode='paste')
        print("'earthengine authenticate'를 실행해주세요.")
        return False


def stitch_images(rgb_img: Image.Image, ir_img: Image.Image) -> Image.Image:
    """RGB와 IR 이미지를 좌우로 이어붙여 하나의 이미지로 만듭니다."""
    h = max(rgb_img.height, ir_img.height)
    
    # 높이를 맞추기 위해 리사이즈
    rgb_img = rgb_img.resize((int(rgb_img.width * h / rgb_img.height), h))
    ir_img = ir_img.resize((int(ir_img.width * h / ir_img.height), h))
    
    total_width = rgb_img.width + ir_img.width
    combined_img = Image.new('RGB', (total_width, h))
    
    # 이미지 붙이기
    combined_img.paste(rgb_img, (0, 0))
    combined_img.paste(ir_img, (rgb_img.width, 0))
    
    # 라벨 추가
    try:
        draw = ImageDraw.Draw(combined_img)
        font = ImageFont.load_default()
    except Exception as e:
        print(f"Warning: 라벨 추가 중 오류: {e}")
    
    return combined_img


def check_data_availability(
    date_str: str,  # 형식: 'YYYY-MM-DD'
    hour: int,  # 시간 (0-23)
    bbox: Tuple[float, float, float, float],  # (west, south, east, north)
) -> dict:
    """
    특정 시간대와 영역에 MODIS 데이터가 있는지 확인합니다.
    
    Args:
        date_str: 날짜 문자열 (YYYY-MM-DD)
        hour: 시간 (0-23)
        bbox: (west, south, east, north) 좌표
    
    Returns:
        {'modis_available': bool, 'cli2text_available': bool, 'satellite': str}
    """
    
    try:
        # 시간대별 위성 선택
        # Terra: 10-11시 (주간)
        # Aqua: 13-14시 (주간) 
        # Terra: 22-23시 (야간)
        # Aqua: 1-2시 (야간)
        
        modis_available = False
        satellite = None
        
        if hour in [10, 11]:
            satellite = 'terra'
            modis_available = True
        elif hour in [13, 14]:
            satellite = 'aqua'
            modis_available = True
        elif hour in [22, 23]:
            satellite = 'terra'
            modis_available = True
        elif hour in [1, 2]:
            satellite = 'aqua'
            modis_available = True
        
        # MODIS 가용성 확인 시 실제 데이터 존재 여부 체크
        if modis_available:
            try:
                west, south, east, north = bbox
                region_box = ee.Geometry.Rectangle([west, south, east, north])
                time_start = ee.Date(date_str)
                time_end = time_start.advance(1, 'day')
                
                # 위성별 컬렉션 선택
                if satellite == 'terra':
                    collection = ee.ImageCollection('MODIS/061/MOD11A1')
                else:
                    collection = ee.ImageCollection('MODIS/061/MYD11A1')
                
                # 데이터 존재 여부 확인
                filtered = collection.filterDate(time_start, time_end).filterBounds(region_box)
                count = filtered.size().getInfo()
                
                if count == 0:
                    modis_available = False
                    
            except Exception as e:
                print(f"데이터 확인 중 오류: {e}")
                modis_available = False
        
        # WeatherQA 데이터 가용성 확인
        weatherqa_info = check_weatherqa_availability(date_str, hour)
        cli2text_available = weatherqa_info['available']
        
        return {
            'modis_available': modis_available,
            'cli2text_available': cli2text_available,
            'satellite': satellite if modis_available else None,
            'weatherqa_image_count': weatherqa_info['image_count'],
            'weatherqa_types': weatherqa_info['image_types']
        }
        
    except Exception as e:
        print(f"데이터 가용성 확인 중 오류: {e}")
        return {
            'modis_available': False,
            'cli2text_available': False,
            'satellite': None,
            'weatherqa_image_count': 0,
            'weatherqa_types': []
        }


def fetch_modis_images(
    date_str: str,  # 형식: 'YYYY-MM-DD'
    satellite: str,  # 'terra' 또는 'aqua'
    bbox: Tuple[float, float, float, float],  # (west, south, east, north)
    is_daytime: bool = True,
    output_dir: str = './assets',
    image_size: Tuple[int, int] = (256, 256),
    empty_threshold: float = 0.5,
    pinpoint: Optional[Tuple[float, float]] = None  # (lat, lng)
) -> Optional[str]:
    """
    MODIS 데이터를 가져와 RGB + LST 합성 이미지를 생성합니다.
    
    Args:
        date_str: 날짜 문자열 (YYYY-MM-DD)
        satellite: 'terra' 또는 'aqua'
        bbox: (west, south, east, north) 좌표
        is_daytime: 주간(True) 또는 야간(False)
        output_dir: 저장 디렉토리
        image_size: 출력 이미지 크기 (width, height)
        empty_threshold: 빈 픽셀 비율 임계값
        pinpoint: 핀포인트 좌표 (lat, lng), 있으면 보라색 점 표시
    
    Returns:
        저장된 파일 경로 또는 None (실패 시)
    """
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 좌표 파싱
    west, south, east, north = bbox
    
    # Earth Engine 지오메트리 생성
    region_box = ee.Geometry.Rectangle([west, south, east, north])
    
    # 날짜 파싱 및 시간 범위 설정
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        time_start = ee.Date(date_str)
        time_end = time_start.advance(1, 'day')
    except Exception as e:
        print(f"날짜 파싱 오류: {e}")
        return None
    
    # MODIS 컬렉션 선택
    if satellite.lower() == 'terra':
        rgb_collection = ee.ImageCollection('MODIS/061/MOD09GA')
        lst_collection = ee.ImageCollection('MODIS/061/MOD11A1')
        sat_name = 'terra'
    elif satellite.lower() == 'aqua':
        rgb_collection = ee.ImageCollection('MODIS/061/MYD09GA')
        lst_collection = ee.ImageCollection('MODIS/061/MYD11A1')
        sat_name = 'aqua'
    else:
        print(f"지원하지 않는 위성: {satellite}")
        return None
    
    time_period = 'day' if is_daytime else 'night'
    
    try:
        # RGB 이미지 처리 (주간/야간 모두)
        rgb_img = None
        print("RGB 이미지 가져오는 중...")
        rgb_filtered = rgb_collection.filterDate(time_start, time_end).filterBounds(region_box)
        rgb_mosaic = rgb_filtered.mean()
        
        rgb = rgb_mosaic.select(
            ['sur_refl_b01', 'sur_refl_b04', 'sur_refl_b03'],
            ['RGB_B1_Red', 'RGB_B4_Green', 'RGB_B3_Blue']
        ).multiply(0.0001)
        
        rgb_vis = rgb.visualize(
            bands=['RGB_B1_Red', 'RGB_B4_Green', 'RGB_B3_Blue'],
            min=0,
            max=1
        )
        
        rgb_url = rgb_vis.getThumbURL({
            'region': region_box.getInfo()['coordinates'],
            'dimensions': list(image_size),
            'format': 'png'
        })
        
        # 이미지 다운로드
        response = requests.get(rgb_url, timeout=30)
        if response.status_code == 200:
            rgb_img = Image.open(io.BytesIO(response.content))
            
            # 빈 픽셀 체크
            img_array = np.array(rgb_img)
            if len(img_array.shape) == 3:
                black_pixels = np.all(img_array == 0, axis=2)
                empty_ratio = np.sum(black_pixels) / (img_array.shape[0] * img_array.shape[1])
            else:
                empty_ratio = np.sum(img_array == 0) / img_array.size
            
            if empty_ratio > empty_threshold:
                print(f"경고: RGB 이미지의 {empty_ratio*100:.1f}%가 빈 픽셀입니다.")
                rgb_img = None
        else:
            print(f"RGB 다운로드 실패: HTTP {response.status_code}")
        
        # LST/IR 이미지 처리 (주/야간 모두)
        print("LST/IR 이미지 가져오는 중...")
        lst_filtered = lst_collection.filterDate(time_start, time_end).filterBounds(region_box)
        lst_mosaic = lst_filtered.mean()
        
        if is_daytime:
            lst = lst_mosaic.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('IR_LST_Celsius')
        else:
            lst = lst_mosaic.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('IR_LST_Celsius')
        
        lst_vis = lst.visualize(
            min=-20,
            max=40,
            palette=['blue', 'cyan', 'green', 'yellow', 'red']
        )
        
        lst_url = lst_vis.getThumbURL({
            'region': region_box.getInfo()['coordinates'],
            'dimensions': list(image_size),
            'format': 'png'
        })
        
        response = requests.get(lst_url, timeout=30)
        if response.status_code != 200:
            print(f"LST 다운로드 실패: HTTP {response.status_code}")
            return None
        
        lst_img = Image.open(io.BytesIO(response.content))
        
        # 빈 픽셀 체크
        img_array = np.array(lst_img)
        if len(img_array.shape) == 3:
            black_pixels = np.all(img_array == 0, axis=2)
            empty_ratio = np.sum(black_pixels) / (img_array.shape[0] * img_array.shape[1])
        else:
            empty_ratio = np.sum(img_array == 0) / img_array.size
        
        if empty_ratio > empty_threshold:
            print(f"경고: LST 이미지의 {empty_ratio*100:.1f}%가 빈 픽셀입니다.")
        
        # 핀포인트가 있으면 RGB와 LST 이미지에 각각 보라색 점 그리기
        if pinpoint:
            try:
                lat, lng = pinpoint
                
                # 경도 -> X 좌표 비율 (0부터 1까지)
                lon_percent = (lng - west) / (east - west)
                # 위도 -> Y 좌표 비율 (위도는 위에서 아래로, 0부터 1까지)
                lat_percent_from_top = (north - lat) / (north - south)
                
                # RGB 이미지에 점 그리기
                if rgb_img:
                    rgb_width, rgb_height = rgb_img.size
                    rgb_x = lon_percent * (rgb_width - 1)
                    rgb_y = lat_percent_from_top * (rgb_height - 1)
                    
                    if 0 <= rgb_x < rgb_width and 0 <= rgb_y < rgb_height:
                        rgb_img = rgb_img.convert('RGB')
                        draw = ImageDraw.Draw(rgb_img)
                        radius = 3
                        draw.ellipse(
                            (rgb_x - radius, rgb_y - radius, rgb_x + radius, rgb_y + radius),
                            fill='purple',
                            outline='purple'
                        )
                        print(f"RGB 핀포인트 표시: ({lat}, {lng}) -> 픽셀 ({rgb_x:.1f}, {rgb_y:.1f})")
                
                # LST 이미지에 점 그리기
                lst_width, lst_height = lst_img.size
                lst_x = lon_percent * (lst_width - 1)
                lst_y = lat_percent_from_top * (lst_height - 1)
                
                if 0 <= lst_x < lst_width and 0 <= lst_y < lst_height:
                    lst_img = lst_img.convert('RGB')
                    draw = ImageDraw.Draw(lst_img)
                    radius = 3
                    draw.ellipse(
                        (lst_x - radius, lst_y - radius, lst_x + radius, lst_y + radius),
                        fill='purple',
                        outline='purple'
                    )
                    print(f"LST 핀포인트 표시: ({lat}, {lng}) -> 픽셀 ({lst_x:.1f}, {lst_y:.1f})")
                    
            except Exception as e:
                print(f"핀포인트 그리기 중 오류: {e}")
        
        # 이미지 합성 (항상 RGB + LST)
        if rgb_img:
            combined_img = stitch_images(rgb_img, lst_img)
        else:
            print("경고: RGB 이미지를 가져올 수 없어 LST만 사용합니다.")
            combined_img = lst_img
        
        # 파일명 생성
        date_formatted = date_obj.strftime('%Y%m%d')
        filename = f"{sat_name}_{time_period}_{date_formatted}_{west:.2f}_{south:.2f}_{east:.2f}_{north:.2f}.png"
        output_path = os.path.join(output_dir, filename)
        
        # 이미지 저장
        combined_img.save(output_path)
        print(f"✓ 이미지 저장 완료: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"MODIS 이미지 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """메인 실행 함수 - 예시"""
    
    # Earth Engine 초기화
    if not initialize_earth_engine():
        return
    
    # 예시 파라미터
    date_str = "2023-04-14"
    satellite = "terra"  # 'terra' 또는 'aqua'
    
    # 미국 본토 영역 (XYXY 형식)
    bbox = (-125.0, 24.0, -66.0, 49.0)  # (west, south, east, north)
    
    # 또는 더 작은 영역 (예: 캘리포니아)
    # bbox = (-124.0, 32.5, -114.0, 42.0)
    
    is_daytime = True  # 주간 이미지
    
    # 이미지 가져오기
    result_path = fetch_modis_images(
        date_str=date_str,
        satellite=satellite,
        bbox=bbox,
        is_daytime=is_daytime,
        output_dir='./assets',
        image_size=(512, 512)
    )
    
    if result_path:
        print(f"\n성공! 이미지가 저장되었습니다: {result_path}")
    else:
        print("\n실패: 이미지를 가져오지 못했습니다.")


if __name__ == "__main__":
    main()
