import torch
import torch.nn as nn
from pathlib import Path
import logging
from src.models import PerceiverEncoder
from train_standalone_cls import StandaloneImageClassifier 

class Stage1InferencePipeline(nn.Module):
    """
    'A' 방식 (별도 분류기 + 인코더)으로 Stage 1 추론을 수행하는 모듈.
    미리 훈련된 분류기와 인코더를 로드하고 '동결(freeze)'합니다.
    """
    def __init__(
        self,
        # 1. 인코더 설정 (train_stage1_mae와 동일해야 함)
        patch_size: int,
        latent_dim: int,
        num_latents: int,
        num_blocks: int,
        num_heads: int,
        moe_num_experts: int,
        num_conditions: int, # cond_map의 크기
        
        # 2. 분류기 설정
        classifier_backbone: str,
        
        # 3. 훈련된 가중치 경로
        encoder_ckpt_path: str,
        classifier_ckpt_path: str,
        
        # 4. 인코더 모드 (Stage 1과 동일)
        encoder_return_all: bool = False
    ):
        super().__init__()
        self.encoder_return_all = encoder_return_all

        # 1. 인코더 로드 및 동결
        logging.info(f"Loading Encoder from {encoder_ckpt_path}")
        self.encoder = PerceiverEncoder(
            patch_size=patch_size,
            latent_dim=latent_dim,
            num_latents=num_latents,
            num_blocks=num_blocks,
            num_heads=num_heads,
            moe_num_experts=moe_num_experts,
            num_conditions=num_conditions,
        )
        self.encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location="cpu"))
        self.encoder.eval()
        self.encoder.requires_grad_(False) # 추론용이므로 동결

        # 2. 분류기 로드 및 동결
        logging.info(f"Loading Standalone Classifier from {classifier_ckpt_path}")
        self.classifier = StandaloneImageClassifier(
            num_classes=num_conditions,
            backbone_name=classifier_backbone
        )
        self.classifier.load_state_dict(torch.load(classifier_ckpt_path, map_location="cpu"))
        self.classifier.eval()
        self.classifier.requires_grad_(False) # 추론용이므로 동결

    @torch.no_grad() # (중요) 이 모듈은 추론 전용이므로 그래디언트가 필요 없음
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        'A' 방식 파이프라인을 실행합니다.
        Args:
            imgs (torch.Tensor): (B, C, H, W) 원본 이미지
        Returns:
            torch.Tensor: (B, D) 또는 (B, L, D) 형태의 lat 텐서
        """
        
        # 1. (A 방식) 원본 이미지로 cond_id 예측
        predicted_logits = self.classifier(imgs)
        predicted_cond_ids = predicted_logits.argmax(dim=-1) # (B,)
        
        # (참고: Stage 1 훈련 시 사용한 마스킹은 '추론' 시에는
        #  일반적으로 적용하지 않지만, MAE의 경우 원본(unmasked) 이미지를
        #  그대로 인코더에 넣는 것이 표준입니다.)
        # masked = random_mask_images(imgs, mask_ratio=0.0) # 마스킹 안 함
        
        # 2. (C 방식) 예측된 cond_id를 사용해 인코더 실행
        lat = self.encoder(
            imgs, # 마스킹되지 않은 원본 이미지
            return_all=self.encoder_return_all, 
            cond_ids=predicted_cond_ids
        )
        
        return lat, predicted_cond_ids