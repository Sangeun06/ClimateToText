"""
WeatherQA 이미지 추론을 위한 간소화된 모듈
inference_simple.py를 기반으로 DeepWeatherInsight용으로 단순화
"""

import os
from typing import List, Dict, Optional
from pathlib import Path

import torch
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image


class WeatherQAInferenceEngine:
    """WeatherQA 이미지 추론 엔진"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        base_model_path: str,
        stage1_encoder_ckpt: str,
        stage1_classifier_ckpt: str,
        device: str = "cuda:1",
        image_size: int = 224,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        
        # Transform 설정
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # 모델 로드
        print(f"Loading WeatherQA inference model on {self.device}...")
        self._load_model(checkpoint_dir, base_model_path, stage1_encoder_ckpt, stage1_classifier_ckpt)
        print("WeatherQA inference model loaded successfully.")
    
    def _load_model(self, checkpoint_dir: str, base_model_path: str, encoder_ckpt: str, classifier_ckpt: str):
        """모델 및 토크나이저 로드"""
        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # LLM 로드
        base_llm = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        llm = PeftModel.from_pretrained(base_llm, checkpoint_dir)
        llm.to(self.device)
        
        # Stage1 Image Encoder 로드
        from stage1_stage2_integration import (
            Stage1ImageEncoderForStage2,
            ImageToTextModelStage1,
        )
        
        image_encoder = Stage1ImageEncoderForStage2(
            encoder_ckpt_path=encoder_ckpt,
            classifier_ckpt_path=classifier_ckpt,
            device=self.device,
            freeze=True,
        )
        
        # Finetuned encoder가 있으면 로드
        finetuned_enc_path = os.path.join(checkpoint_dir, "stage1_encoder_finetuned.pt")
        if os.path.exists(finetuned_enc_path) and hasattr(image_encoder, "pipeline") and hasattr(image_encoder.pipeline, "encoder"):
            try:
                enc_state = torch.load(finetuned_enc_path, map_location=self.device, weights_only=True)
            except TypeError:
                enc_state = torch.load(finetuned_enc_path, map_location=self.device)
            try:
                image_encoder.pipeline.encoder.load_state_dict(enc_state, strict=False)
                print(f"Loaded finetuned Stage1 encoder from {finetuned_enc_path}")
            except Exception as e:
                print(f"Warning: failed to load finetuned Stage1 encoder: {e}")
        
        # 통합 모델
        self.model = ImageToTextModelStage1(
            llm=llm,
            image_encoder=image_encoder,
            num_image_tokens=4,
        )
        self.model.to(self.device)
        self.model.eval()
    
    def infer_single(
        self,
        image_path: str,
        cond_name: str,
        prompt: str = "Forecast summary in 2-3 sentences. Focus on hazards, region, and timing.",
        max_new_tokens: int = 80,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 4,
        repetition_penalty: float = 1.1,
    ) -> str:
        """단일 이미지에 대한 추론"""
        try:
            # 이미지 로드 및 전처리
            img = Image.open(image_path).convert("RGB")
            pixel_values = self.transform(img).unsqueeze(0).to(self.device)
            
            # 프롬프트 생성
            full_prompt = f"Condition: {cond_name}\n{prompt}"
            
            # 추론
            cond_ids = torch.zeros(pixel_values.size(0), dtype=torch.long, device=self.device)
            with torch.no_grad():
                output = self.model.generate(
                    pixel_values=pixel_values,
                    cond_ids=cond_ids,
                    tokenizer=self.tokenizer,
                    generation_prompt=full_prompt,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=repetition_penalty,
                )[0]
            
            return output
        
        except Exception as e:
            print(f"Error inferring {image_path}: {e}")
            return f"[Inference Error: {str(e)}]"
    
    def infer_batch(
        self,
        image_infos: List[Dict[str, str]],
        prompt: str = "Forecast summary in 2-3 sentences. Focus on hazards, region, and timing.",
        max_new_tokens: int = 128,
    ) -> List[Dict[str, str]]:
        """
        여러 이미지에 대한 배치 추론
        
        Args:
            image_infos: [{'type': 'shr6', 'path': '/path/to/img.gif', ...}, ...]
        
        Returns:
            [{'type': 'shr6', 'generated': 'text...', ...}, ...]
        """
        results = []
        for info in image_infos:
            generated = self.infer_single(
                image_path=info['path'],
                cond_name=info['type'],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            results.append({
                **info,
                'generated': generated,
            })
        return results


# 글로벌 인스턴스 (lazy loading)
_inference_engine: Optional[WeatherQAInferenceEngine] = None


def get_inference_engine() -> WeatherQAInferenceEngine:
    """싱글톤 패턴으로 추론 엔진 반환"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = WeatherQAInferenceEngine(
            checkpoint_dir="/home/agi592/twkim/checkpoints/stage2_summary_centric_ep2_wu_norm_fixed_512",
            base_model_path="/home/agi592/models/mistral-7b",
            stage1_encoder_ckpt="/home/agi592/kse/ClimateToText/stage1_curriculum_runs/step1_all_types/stage1_vision_encoder_mae.pt",
            stage1_classifier_ckpt="/home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt",
            device="cuda:1",
            image_size=224,
        )
    return _inference_engine


def infer_weatherqa_images(image_infos: List[Dict]) -> List[Dict]:
    """
    WeatherQA 이미지들에 대한 추론 수행
    
    Args:
        image_infos: get_weatherqa_images()의 반환값
    
    Returns:
        각 이미지에 'generated' 필드가 추가된 리스트
    """
    if not image_infos:
        return []
    
    try:
        engine = get_inference_engine()
        return engine.infer_batch(image_infos)
    except Exception as e:
        print(f"WeatherQA inference error: {e}")
        # 에러 발생 시 빈 generated 반환
        for info in image_infos:
            info['generated'] = f"[Inference unavailable: {str(e)}]"
        return image_infos
