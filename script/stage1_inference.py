import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.models import PerceiverEncoder, ResNetEncoder, ConditionalPerceiverEncoder
from .train_standalone_cls import StandaloneImageClassifier
from .train_image_encoder import _build_stage1_dataloader, setup_logging, set_seed


# ROOT = Path(__file__).resolve().parent.parent

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
        encoder_return_all: bool = False,
        encoder_mode: str = "perceiver_patch_mae",
    ):
        super().__init__()
        self.encoder_mode = encoder_mode
        self.encoder_return_all = encoder_return_all

        # 1. 인코더 로드 및 동결
        logging.info(f"Loading Encoder (mode={encoder_mode}) from {encoder_ckpt_path}")

        if encoder_mode == "resnet":
            self.encoder = ResNetEncoder(
                pretrained=True,
                output_dim=latent_dim,
            )
        elif encoder_mode == "perceiver_patch_mae":
            self.encoder = ConditionalPerceiverEncoder(
                num_types=num_conditions,
                patch_size=patch_size,
                latent_dim=latent_dim,
                num_latents=num_latents,
                num_blocks=num_blocks,
                num_heads=num_heads,
                moe_num_experts=moe_num_experts,
            )
        else:
            # 기본값: PerceiverEncoder (Stage1 MAE)
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
        if self.encoder_mode == "perceiver":
            # PerceiverEncoder: (B,C,H,W) + cond_ids
            lat = self.encoder(
                imgs,  # 마스킹되지 않은 원본 이미지
                return_all=self.encoder_return_all,
                cond_ids=predicted_cond_ids,
            )
        elif self.encoder_mode == "perceiver_patch_mae":
            # ConditionalPerceiverEncoder: (이미지, type_ids, keep_ratio)
            lat_seq, _, _ = self.encoder(
                imgs,
                type_ids=predicted_cond_ids,
                keep_ratio=1.0,  # 추론에서는 전체 패치를 사용
            )
            lat = lat_seq if self.encoder_return_all else lat_seq.mean(dim=1)
        elif self.encoder_mode == "resnet":
            # ResNetEncoder: (B,C,H,W) -> (B,L,D)
            lat_seq = self.encoder(imgs)
            lat = lat_seq if self.encoder_return_all else lat_seq.mean(dim=1)
        else:
            # 정의되지 않은 모드: 단순 호출
            lat = self.encoder(imgs)
        
        return lat, predicted_cond_ids


if __name__ == "__main__":
    """
    간단한 Stage1InferencePipeline 사용 예제 (하드코딩된 기본값).
    """
    setup_logging()
    seed = 42
    set_seed(seed)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. 하드코딩된 데이터로더 설정
    class DummyArgs:
        pass

    args = DummyArgs()
    # train_image_encoder.py 의 기본값과 동일하게 설정
    args.stage1_sources = "weatherqa"
    args.chatearthnet_json_path = ""
    args.chatearthnet_image_root = ""
    args.climateiqa_json_path = ""
    args.climateiqa_image_root = ""
    args.weatherqa_json_path = "/home/agi592/kse/ClimateToText/data/WeatherQA/dataset_2014-2020.json"
    args.weatherqa_image_dir = "/home/agi592/kse/ClimateToText/data/WeatherQA/WeatherQA_MD_2014-2019"
    # args.weatherqa_include_types = ""
    args.weatherqa_include_types = "thea,mcon,ttd"
    args.pretrain_years = "2019"
    args.image_size = 224
    args.batch_size = 8
    args.num_workers = 4
    args.seed = seed
    # 기타 Stage2/3 관련 필드는 dataloader에 필요 없음

    logging.info("Building Stage1 dataloader for inference (hardcoded args)...")
    dl, cond_map = _build_stage1_dataloader(args)
    num_conditions = len(cond_map)
    logging.info(f"Loaded condition map with {num_conditions} types.")

    # 2. 파이프라인 구성 (하드코딩된 경로와 설정)
    encoder_ckpt_path = Path("/home/agi592/kse/ClimateToText/stage1_mtl_weatherqa_three_patch/stage1_vision_encoder_mae.pt")
    classifier_ckpt_path = Path("/home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt")

    logging.info("Initializing Stage1InferencePipeline (hardcoded config)...")
    pipeline = Stage1InferencePipeline(
        patch_size=16,
        latent_dim=768,
        num_latents=196,
        num_blocks=6,
        num_heads=8,
        moe_num_experts=8,
        num_conditions=num_conditions,
        classifier_backbone="efficientnet_b0",
        encoder_ckpt_path=str(encoder_ckpt_path),
        classifier_ckpt_path=str(classifier_ckpt_path),
        encoder_return_all=False,
        encoder_mode="perceiver_patch_mae",
    ).to(device)

    pipeline.eval()

    # 3. 예시 추론 실행 (한 두 배치만)
    logging.info("Running inference on a few batches for demo (hardcoded)...")
    
    max_batches = 1
    for bi, batch in enumerate(dl):
        if bi >= max_batches:
            break
        imgs = batch["images"].to(device)

        with torch.no_grad():
            latents, pred_cond_ids = pipeline(imgs)

        logging.info(
            f"[Batch {bi}] imgs={imgs.shape}, "
            f"latents={tuple(latents.shape)}, "
            f"pred_cond_ids={tuple(pred_cond_ids.shape)}"
        )

    logging.info("Stage1InferencePipeline hardcoded demo finished.")
