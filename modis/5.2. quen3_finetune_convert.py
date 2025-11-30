import json
import os
from typing import List, Dict

# 사용자 정의 모듈 import (경로가 맞는지 확인하세요)
from quen_loader import load_weather_dataset

# --- 설정 ---
anno_loc = "generated_weather_dataset_gemini.jsonl"
combined_dir = "./combined/"
thinks_loc = "finetune_thinkings.jsonl"  # think_answer가 들어있는 JSONL 파일
output_file = "5_weather_train_swift.jsonl"

# 당신이 정의한 시스템 프롬프트 (학습 데이터에 포함시켜야 모델이 이 역할을 학습함)
SYSTEM_PROMPT = """You are an expert meteorologist and a senior data scientist. Your critical task is to answer the given question based on the provided satellite imagery.

You will be given:
1. A combined satellite image (RGB on the left, LST on the right, with a purple dot indicating the observation location).
2. A Question about the imagery.

## Instructions for Your Response
* Think step-by-step about the observation, but keep your thinking process CONCISE (under 1500 tokens).
* Your final answer must be 4-6 sentences long, professional, confident, and directly address the question."""

# 템플릿 포맷 (질문 앞뒤 텍스트)
USER_PROMPT_TEMPLATE = """## Your Task

Based on the provided satellite imagery, answer the following question:

{question}"""

def convert_to_swift_format():
    print(">>> 데이터셋 로딩 중...")
    full_dataset, train_dataset, eval_dataset = load_weather_dataset(
        anno_loc=anno_loc,
        combined_dir=combined_dir,
        test_size=0.1,
        seed=42,
        verbose=True
    )
    
    # thinks_loc에서 think_answer를 id를 key로 하는 dict로 로드
    print(f">>> {thinks_loc}에서 think_answer 로딩 중...")
    think_answers = {}
    with open(thinks_loc, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if 'id' in item and 'think_answer' in item:
                    think_answers[item['id']] = item['think_answer']
            except json.JSONDecodeError:
                continue
    
    print(f">>> {len(think_answers)}개의 think_answer를 로드했습니다.")
    print(f">>> 변환 시작 (Train set size: {len(train_dataset)})")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        skipped_count = 0
        for sample in train_dataset:
            # 1. 이미지 경로 수집
            image_paths = sample.get('images', [])
            if not image_paths or not os.path.exists(image_paths[0]):
                print(f"Skipping sample {sample['id']}: Image not found")
                continue
            
            # 2. ID에 해당하는 think_answer 가져오기
            sample_id = sample.get('id')
            if sample_id not in think_answers:
                print(f"Skipping sample {sample_id}: No think_answer found")
                skipped_count += 1
                continue
            
            assistant_msg = think_answers[sample_id]
            
            # 3. 대화 내용 파싱
            # 기존 데이터셋 구조: conversations -> [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'}]
            user_msg = ""
            
            for conv in sample['conversations']:
                if conv['from'] == 'human':
                    # <image> 태그 제거 후 템플릿 적용
                    raw_q = conv['value'].replace("<image>", "").strip()
                    user_msg = USER_PROMPT_TEMPLATE.format(question=raw_q)
            
            if not user_msg:
                continue

            # 3. ms-swift 포맷으로 구성
            # Qwen3-VL은 이미지 처리를 위해 <image> 태그가 필요할 수 있으나,
            # ms-swift는 'images' 키가 있으면 자동으로 처리해줍니다. 
            # 단, 명시적으로 content에 넣는 것이 안전합니다.
            
            swift_entry = {
                "images": image_paths,  # 이미지 경로 리스트
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": "<image>" + user_msg # 이미지 토큰을 질문 앞에 배치
                    },
                    {
                        "role": "assistant",
                        "content": assistant_msg
                    }
                ]
            }
            
            f.write(json.dumps(swift_entry, ensure_ascii=False) + '\n')
        
        if skipped_count > 0:
            print(f">>> {skipped_count}개의 샘플이 think_answer가 없어서 스킵되었습니다.")
            
    print(f">>> 변환 완료: {output_file}")

if __name__ == "__main__":
    convert_to_swift_format()
    