import json
import os
import time
from PIL import Image
from typing import Dict
import torch

# quen_loader 모듈에서 함수 import
from quen_loader import load_weather_dataset

# --- 1. 입력 설정 ---
anno_loc = "generated_weather_dataset_gemini.jsonl"
combined_dir = "./combined/"

# 데이터셋 로드
full_dataset, train_dataset, eval_dataset = load_weather_dataset(
    anno_loc=anno_loc,
    combined_dir=combined_dir,
    test_size=0.1,
    seed=42,
    verbose=True
)

# --- 확인용 샘플 출력 ---
print("\n--- 훈련 세트 샘플 (1개) ---")
print(train_dataset[0])

#######################################################################################

import requests
from sglang import Engine
from transformers import AutoProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 함수 정의나 import는 블록 밖에 있어도 됩니다.

SYSTEM_PROMPT = """You are a Senior Satellite Meteorologist and Data Scientist.
Your task is to generate a logical, step-by-step "Chain of Thought" (<think>...</think>) that leads to a specific conclusion based on satellite imagery (RGB and LST).

## Your Goal
You will be provided with:
1. Satellite Images
2. A User Question
3. The Ground Truth Answer (The correct conclusion)

You must write the internal thinking process that a professional meteorologist would have *before* arriving at that Ground Truth Answer.

## Critical Constraints (DO NOT FAIL THESE)
1. **No Leakage:** NEVER mention that you were provided with the "Ground Truth Answer". Do not use phrases like "The answer says...", "To match the truth...", or "As provided".
2. **First-Person Discovery:** Write as if you are analyzing the image for the first time. Use phrases like "I observe...", "The LST indicates...", "This pattern suggests...".
3. **Visual Evidence First:** Always start by grounding your thought in specific visual features (e.g., "The deep purple area in the LST...", "The textured white clouds in the RGB...").
4. **Handle Contradictions:** If the User Question contains a false premise (Theme 4), your thought process should explicitly identify the mismatch between the question and the image evidence.
"""

USER_PROMPT_TEMPLATE = """## Input Data
**User Question:** {question}

**Target Conclusion (Ground Truth):** {answer}

## Instruction
Generate the thinking process (<think> ... </think>) that logically leads to the "Target Conclusion".
The thought process must:
1. Analyze the relevant visual features in the images.
2. Apply meteorological knowledge to interpret those features.
3. Address the specific type of question asked (e.g., if it asks "Why", focus on causes; if it asks "What if", focus on simulation).
4. Naturally conclude with the sentiment or facts found in the Target Conclusion.

**Start your response immediately with <think>.**
"""


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
            answer_content = ""
            for conv in sample['conversations']:
                if conv['from'] == 'gpt':
                    answer_content = conv['value']
                    break
            formatted_question = USER_PROMPT_TEMPLATE.format(question=question_text, answer=answer_content)
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
    # 1. 모델 경로 및 프로세서 로드
    checkpoint_path = "Qwen/Qwen3-VL-8B-Thinking"
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    # 2. train_dataset 전체 사용
    num_samples = len(train_dataset)
    print(f"\n--- 전체 train set({num_samples}개 샘플)로 추론 시작 ---")
    
    output_file = "finetune_thinkings.jsonl"
    
    # 3. SGLang 엔진 초기화
    llm = Engine(
        model_path=checkpoint_path,
        enable_multimodal=True,
        tp_size=1,
        mem_fraction_static=0.8,
        context_length=4096, 
    )

    # 4. 샘플링 파라미터
    think_max_tokens = 1500
    sampling_params_think = {
        "max_new_tokens": think_max_tokens,
        "temperature": 0.7,
        "stop": ["</think>"],
    }

    # --- [수정됨] 배치 처리 로직 ---
    BATCH_SIZE = 50
    
    for i in range(0, num_samples, BATCH_SIZE):
        # 현재 배치 구간의 끝 인덱스 계산
        end_idx = min(i + BATCH_SIZE, num_samples)
        
        print(f"\nRunning Batch: {i} ~ {end_idx}")
        
        # [핵심 수정] 슬라이싱(dataset[i:j]) 대신 인덱싱 루프를 사용해야 
        # 딕셔너리의 리스트([{}, {}, ...])를 얻을 수 있습니다.
        batch_samples = [train_dataset[idx] for idx in range(i, end_idx)]
        
        # 배치용 리스트 준비
        batch_prompts = []
        batch_images = []
        valid_sample_indices = [] # 로드 성공한 샘플이 batch_samples 내에서 몇 번째인지 저장

        # 배치 데이터 전처리
        for local_idx, sample in enumerate(batch_samples):
            try:
                # 1. 이미지 로드
                image_inputs = []
                has_valid_image = True
                
                # sample은 이제 확실히 Dictionary입니다.
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
                valid_sample_indices.append(local_idx) # 현재 배치의 몇 번째 샘플인지 저장

            except Exception as e:
                # KeyError 뿐만 아니라 다른 에러도 잡아서 멈추지 않게 함
                print(f"Error 발생 (ID: {sample.get('id', 'unknown')}): {e}")
                continue

        if not batch_prompts:
            continue

        # --- SGLang 병렬 실행 ---
        start = time.time()
        try:
            batch_outputs = llm.generate(
                prompt=batch_prompts,
                image_data=batch_images,
                sampling_params=sampling_params_think
            )
        except Exception as e:
            print(f"SGLang Generate Error: {e}")
            continue
            
        elapsed_time = time.time() - start
        avg_time = elapsed_time / len(batch_prompts)

        # 결과 저장 로직
        new_lines = []
        for k, output in enumerate(batch_outputs):
            # valid_sample_indices를 통해 원본 샘플 매칭
            original_sample = batch_samples[valid_sample_indices[k]]
            
            # 텍스트 추출
            raw_think_output = output['text'] if isinstance(output, dict) else output.text
            
            # 태그 보정
            current_think_text = raw_think_output.strip()
            if not current_think_text.endswith("</think>"):
                current_think_text += "</think>"
            
            clean_think = current_think_text.replace("<think>", "").replace("</think>", "").strip()

            # 정답(Answer) 추출
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

            # 결과 JSON 구성
            result_record = {
                "id": original_sample.get('id'),
                "question": user_question,
                "think_answer": f"<think>\n{clean_think}\n</think>\n\n{answer_content}",
                "inference_time": avg_time
            }
            new_lines.append(json.dumps(result_record, ensure_ascii=False))

        # 파일 쓰기
        with open(output_file, 'a', encoding='utf-8') as f:
            for line in new_lines:
                f.write(line + '\n')

    llm.shutdown()
    print(f"\n완료. {output_file} 확인.")
    
if __name__ == '__main__':
    # 메인 함수를 이 블록 안에서만 실행합니다.
    run_model()