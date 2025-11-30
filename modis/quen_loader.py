"""
Weather Dataset Loader
데이터 로드 및 전처리를 위한 재사용 가능한 모듈
"""

import pandas as pd
from datasets import Dataset, DatasetDict
import json
import os
import re
from PIL import Image, ImageDraw
from typing import Dict, Optional, Tuple


def parse_full_filename_metadata(filename: str) -> Dict:
    """파일 이름의 모든 메타데이터를 파싱하여 딕셔너리로 반환합니다."""
    metadata = {}
    pattern = r'([^_]+)_([^_]+)_(day|night)_coco(\d+)_(\d{8})_lh(\d+)_([-\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)\.png'
    match = re.match(pattern, filename)
    if match:
        metadata['filename'] = filename
        metadata['id'] = match.group(1)
        metadata['satellite'] = match.group(2)
        metadata['day_night'] = match.group(3)
        metadata['coco'] = int(match.group(4))
        metadata['utc_yymmddhh'] = match.group(5)
        metadata['local_hour_filename'] = int(match.group(6))
        metadata['bbox_lat1'] = float(match.group(7))
        metadata['bbox_lon1'] = float(match.group(8))
        metadata['bbox_lat2'] = float(match.group(9))
        metadata['bbox_lon2'] = float(match.group(10))
    return metadata


def calculate_dot_position(metadata: Dict) -> Tuple[Optional[float], Optional[float], bool]:
    """
    메타데이터(CSV의 lat/lon 및 파일명의 bbox)를 기반으로
    128x128 이미지 상의 점의 (x, y) 픽셀 위치를 계산합니다.
    """
    dot_keys = ['lat', 'lon', 'bbox_lat1', 'bbox_lon1', 'bbox_lat2', 'bbox_lon2']
    if not all(key in metadata for key in dot_keys):
        return None, None, False

    try:
        lat_target = float(metadata['lat'])
        lon_target = float(metadata['lon'])
        lat_min = float(metadata['bbox_lat1'])
        lon_min = float(metadata['bbox_lon1'])
        lat_max = float(metadata['bbox_lat2'])
        lon_max = float(metadata['bbox_lon2'])

        img_width = 128
        img_height = 128
        
        if (lon_max - lon_min) == 0 or (lat_max - lat_min) == 0:
            return None, None, False
        
        lon_percent = (lon_target - lon_min) / (lon_max - lon_min)
        pixel_x = lon_percent * (img_width - 1)
        
        lat_percent_from_top = (lat_max - lat_target) / (lat_max - lat_min)
        pixel_y = lat_percent_from_top * (img_height - 1)

        if not (0 <= pixel_x < img_width and 0 <= pixel_y < img_height):
            return None, None, False
        
        return pixel_x, pixel_y, True

    except (ValueError, TypeError, ZeroDivisionError):
        return None, None, False


def draw_purple_dot_on_image(img: Image.Image, pixel_x: float, pixel_y: float) -> Image.Image:
    """이미지에 보라색 점을 그립니다."""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((pixel_x - 2, pixel_y - 2, pixel_x + 2, pixel_y + 2), fill='purple', outline='purple')
    return img


def load_csv_lookup(csv_loc: str) -> Dict:
    """
    CSV 파일을 로드하여 station_id를 키로 하는 lookup 딕셔너리를 생성합니다.
    
    Args:
        csv_loc: CSV 파일 경로
    
    Returns:
        station_id를 키로 하는 딕셔너리 (각 값은 lat/lon을 포함한 딕셔너리 리스트)
    """
    print(f"CSV 파일 로드 중: {csv_loc}")
    csv_df = pd.read_csv(csv_loc)
    
    csv_lookup = {}
    for _, row in csv_df.iterrows():
        station_id = row['id']
        if station_id not in csv_lookup:
            csv_lookup[station_id] = []
        csv_lookup[station_id].append({
            'lat': row['lat'],
            'lon': row['lon']
        })
    
    return csv_lookup


def load_weather_dataset(
    anno_loc: str,
    combined_dir: str,
    test_size: float = 0.1,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Weather Q&A 데이터셋을 로드하고 전처리합니다.
    
    Args:
        anno_loc: JSONL 어노테이션 파일 경로
        combined_dir: Combined 이미지 디렉토리
        test_size: 테스트 세트 비율 (기본값: 0.1)
        seed: 랜덤 시드 (기본값: 42)
        verbose: 진행 상황 출력 여부 (기본값: True)
    
    Returns:
        (full_dataset, train_dataset, eval_dataset) 튜플
    """
    all_data_list = []
    
    if verbose:
        print(f"'{anno_loc}' 파일 처리 시작...")
    
    # JSONL 파일을 한 줄씩 읽어옵니다.
    with open(anno_loc, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 각 줄을 JSON 객체로 파싱
                data = json.loads(line)
                
                sample_id = data.get("sample_id")
                filename = data.get("source_filename")
                
                if not sample_id or not filename:
                    if verbose:
                        print("경고: 'sample_id' 또는 'source_filename'이 없는 줄을 건너뜁니다.")
                    continue
                    
                # 1. Combined 이미지 경로 구성
                combined_path = os.path.join(combined_dir, filename)
                
                # SFTTrainer가 참조할 이미지 경로 (단일 이미지)
                image_paths = [combined_path]
                
                # 2. 'generated_qas' 리스트(7개)를 순회하며 개별 레코드로 분할
                qas = data.get("generated_qas", [])
                for i, qa_pair in enumerate(qas):
                    question = qa_pair.get("question")
                    answer = qa_pair.get("answer")
                    
                    if not question or not answer:
                        continue
                        
                    # 3. SFTTrainer (sharegpt 형식)에 맞는 딕셔너리 생성
                    base_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
                    new_record = {
                        "id": f"{base_filename}_q{i}", 
                        "images": image_paths, 
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<image>\n{question}" 
                            },
                            {
                                "from": "gpt",
                                "value": answer
                            }
                        ]
                    }
                    all_data_list.append(new_record)
                    
            except json.JSONDecodeError:
                if verbose:
                    print(f"경고: JSON 파싱 오류. 해당 줄을 건너뜁니다: {line[:50]}...")
    
    if verbose:
        print(f"JSONL 파일을 처리 완료. 총 {len(all_data_list)}개의 Q&A 레코드를 생성했습니다.")
    
    if not all_data_list:
        raise ValueError("처리할 데이터가 없습니다.")
    
    # DataFrame 및 Dataset으로 변환
    df = pd.DataFrame(all_data_list)
    full_dataset = Dataset.from_pandas(df)
    
    # 데이터셋 분할
    split_dataset_dict = full_dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_dataset_dict['train']
    eval_dataset = split_dataset_dict['test']
    
    if verbose:
        print("\n--- 데이터 로드 및 분할 완료 ---")
        print(f"총 레코드: {len(full_dataset)}")
        print(f"훈련 세트 (train): {len(train_dataset)}개")
        print(f"평가 세트 (eval): {len(eval_dataset)}개")
    
    return full_dataset, train_dataset, eval_dataset
