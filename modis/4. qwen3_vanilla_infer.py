import json
import os
import time
from PIL import Image
from typing import Dict
import torch

# quen_loader 모듈에서 함수 import
from quen_loader import load_weather_dataset

# --- 1. 입력 설정 ---
anno_loc = "generated_weather_dataset_gemini_pureeval.jsonl"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
BATCH_SIZE = 10
SRT_IDX = 0
DATASET_TYPE = "full"  # "full", "train" 또는 "eval"
checkpoint_path="output/qwen3_weather_merged"
# ------------------

combined_dir = "./combined/"
# 데이터셋 로드
full_dataset, train_dataset, eval_dataset = load_weather_dataset(
    anno_loc=anno_loc,
    combined_dir=combined_dir,
    test_size=0.1,
    seed=42,
    verbose=True
)

# 데이터셋 선택
if DATASET_TYPE == "full":
    selected_dataset = full_dataset
elif DATASET_TYPE == "train":
    selected_dataset = train_dataset
elif DATASET_TYPE == "eval":
    selected_dataset = eval_dataset
else:
    raise ValueError(f"Invalid DATASET_TYPE: {DATASET_TYPE}. Must be 'full', 'train', or 'eval'")

# --- 확인용 샘플 출력 ---
print(f"\n--- {DATASET_TYPE.upper()} 세트 샘플 (1개) ---")
print(selected_dataset[0])

#######################################################################################

import requests
from sglang import Engine
from transformers import AutoProcessor

# 함수 정의나 import는 블록 밖에 있어도 됩니다.

SYSTEM_PROMPT = """You are an expert meteorologist and a senior data scientist. Your critical task is to answer the given question based on the provided satellite imagery.

You will be given:
1.  A *combined* satellite image (RGB on the left, LST on the right, with a purple dot indicating the observation location).
2.  A Question about the imagery.

## Instructions for Your Response
* Think step-by-step about the observation, but keep your thinking process CONCISE (under 1500 tokens).
* Your final answer must be 4-6 sentences long, professional, confident, and directly address the question.
"""

USER_PROMPT_TEMPLATE = """## Your Task

Based on the provided satellite imagery, answer the following question:

{question}"""


def convert_conversation_to_messages(sample):
    """train_dataset의 샘플을 Qwen 메시지 형식으로 변환 (system과 user로 분리)"""
    messages = []
    
    for conv in sample['conversations']:
        # human의 질문만 처리
        if conv['from'] == "human":
            # System message 추가
            messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
            
            # User message 구성 (이미지 + 질문)
            user_content = []
            
            # 이미지 추가 (sample['images']에서 가져옴)
            for img_path in sample['images']:
                user_content.append({
                    "type": "image",
                    "image": img_path
                })
            
            # <image> 토큰을 제거한 순수 텍스트만 추가
            question_text = conv['value'].replace("<image>", "").strip()
            formatted_question = USER_PROMPT_TEMPLATE.format(question=question_text)
            user_content.append({
                "type": "text",
                "text": formatted_question
            })
            
            messages.append({
                "role": "user",
                "content": user_content
            })
    
    return messages


def run_model():
    global checkpoint_path
    """메인 실행 로직을 이 함수 안으로 넣습니다."""
    
    # 1. 모델 경로 및 프로세서 로드
    base_model_name = "Qwen/Qwen3-VL-8B-Thinking" 
    
    print(f"\n--- Processor 로딩: {base_model_name} ---")
    # 로컬 경로(checkpoint_path) 대신 원본 이름(base_model_name) 사용
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True) 
    
    print(f"\n--- Model(SGLang) 로딩: {checkpoint_path} ---")
    # checkpoint_path가 없는 경우 기본값 말고 기본 모델 사용, 있는 경우 해당 경로 사용
    output_file = "inference_results_pe.jsonl"
    if checkpoint_path is not None:
        output_file = "inference_results_ft_pe.jsonl"
    else:
        checkpoint_path = base_model_name
    
    # 3. SGLang 엔진 초기화 (context size 확장)
    llm = Engine(
        model_path=checkpoint_path,
        enable_multimodal=True,
        tp_size=1,
        mem_fraction_static=0.8,
        context_length=6144,  # Context window 크기 설정
    )

    # 4. 각 샘플에 대해 추론 실행
    # Thinking과 Answer를 분리하여 생성
    think_max_tokens = 1500  # Thinking 최대 토큰
    answer_max_tokens = 800  # Answer 최대 토큰
    
    sampling_params_think = {
        "max_new_tokens": think_max_tokens,
        "temperature": 0.7,
        "stop": ["</think>"],  # </think> 태그에서 강제 중단
    }
    
    sampling_params_answer = {
        "max_new_tokens": answer_max_tokens,
        "temperature": 0.7,
    }
    
    
    num_samples = len(selected_dataset)
    print(f"\n총 {DATASET_TYPE} 샘플 수: {num_samples}")
    
    for i in range(SRT_IDX, num_samples, BATCH_SIZE):
        # 현재 배치 구간의 끝 인덱스 계산
        end_idx = min(i + BATCH_SIZE, num_samples)
        
        print(f"\n{'='*60}")
        print(f"Running Batch: {i} ~ {end_idx}")
        
        # 배치 샘플 수집
        batch_samples = [selected_dataset[idx] for idx in range(i, end_idx)]
        
        # 배치용 리스트 준비
        batch_prompts = []
        batch_images = []
        valid_sample_indices = []  # 로드 성공한 샘플의 인덱스
        
        # 배치 데이터 전처리
        for local_idx, sample in enumerate(batch_samples):
            try:
                # 1. 이미지 로드
                image_inputs = []
                has_valid_image = True
                
                if 'images' in sample:
                    for img_path in sample['images']:
                        if os.path.exists(img_path):
                            img = Image.open(img_path).convert('RGB')
                            image_inputs.append(img)
                        else:
                            print(f"경고: 이미지 없음 {img_path}")
                            has_valid_image = False
                
                if not image_inputs or not has_valid_image:
                    continue
                
                # 2. 메시지 변환
                messages = convert_conversation_to_messages(sample)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # 리스트에 추가
                batch_prompts.append(text)
                batch_images.append(image_inputs)
                valid_sample_indices.append(local_idx)
                
            except Exception as e:
                print(f"Error 발생 (ID: {sample.get('id', 'unknown')}): {e}")
                continue
        
        if not batch_prompts:
            continue
        
        # --- [Stage 1] Thinking Generation (배치) ---
        print(f"  [Stage 1] Generating Thoughts for {len(batch_prompts)} samples...")
        start = time.time()
        
        try:
            batch_think_outputs = llm.generate(
                prompt=batch_prompts,
                image_data=batch_images,
                sampling_params=sampling_params_think
            )
        except Exception as e:
            print(f"SGLang Generate Error (Thinking): {e}")
            continue
        
        think_elapsed = time.time() - start
        
        # Thinking 결과 처리
        processed_thinks = []
        for k, output in enumerate(batch_think_outputs):
            raw_think_output = output['text'] if isinstance(output, dict) else output.text
            current_think_text = raw_think_output.strip()
            
            if not current_think_text.endswith("</think>"):
                current_think_text += "</think>"
            
            processed_thinks.append(current_think_text)
        
        # --- [Stage 2] Answer Generation (배치) ---
        print(f"  [Stage 2] Generating Answers for {len(batch_prompts)} samples...")
        
        # 각 샘플에 대해 think를 추가한 프롬프트 생성
        answer_prompts = [batch_prompts[k] + processed_thinks[k] for k in range(len(batch_prompts))]
        
        try:
            batch_answer_outputs = llm.generate(
                prompt=answer_prompts,
                image_data=batch_images,
                sampling_params=sampling_params_answer
            )
        except Exception as e:
            print(f"SGLang Generate Error (Answer): {e}")
            continue
        
        total_elapsed = time.time() - start
        avg_time = total_elapsed / len(batch_prompts)
        
        # 결과 저장
        new_lines = []
        for k, answer_output in enumerate(batch_answer_outputs):
            original_sample = batch_samples[valid_sample_indices[k]]
            
            # Answer 텍스트 추출
            infer_content = answer_output['text'] if isinstance(answer_output, dict) else answer_output.text
            infer_content = infer_content.strip()
            
            # Ground truth 추출
            answer_content = ""
            if 'conversations' in original_sample:
                for conv in original_sample['conversations']:
                    if conv['from'] == 'gpt':
                        answer_content = conv['value']
                        break
            
            # User 질문 추출
            user_question = ""
            if 'conversations' in original_sample:
                for conv in original_sample['conversations']:
                    if conv['from'] == 'human':
                        user_question = conv['value'].replace("<image>", "").strip()
                        break
            
            # Think 정리
            clean_think = processed_thinks[k].replace("<think>", "").replace("</think>", "").strip()
            
            # 결과 JSON 구성
            result_record = {
                "id": original_sample.get('id'),
                "question": user_question,
                "think": clean_think,
                "infer": infer_content,
                "answer": answer_content,
                "inference_time": avg_time
            }
            new_lines.append(json.dumps(result_record, ensure_ascii=False))
        
        # 파일 쓰기
        with open(output_file, 'a', encoding='utf-8') as f:
            for line in new_lines:
                f.write(line + '\n')
        
        print(f"  -> Batch 완료. 평균 시간: {avg_time:.2f}s/sample")

    llm.shutdown()
    print(f"\n완료. {output_file} 확인.")

if __name__ == '__main__':
    # 메인 함수를 이 블록 안에서만 실행합니다.
    run_model()