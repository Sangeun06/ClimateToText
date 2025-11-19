from __future__ import annotations

"""
세 단계 VLM 학습 파이프라인 (WeatherQA 중심, 확장 가능 구조)

Stage 1) Masked AutoEncoder(유사)로 비전 인코더 사전학습 (이미지 복원)
Stage 2) LLM 임베딩과 특징 정렬(Alignment) - LLM은 고정, 비전 인코더 + 프로젝션 레이어 학습
Stage 3) LoRA로 LLM 미세조정 + 비전 인코더/프로젝터 지속 학습 (VLM 파인튜닝)

"""

import argparse
import json
import logging
import math
import random
import sys
from tqdm import tqdm

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from torch.utils.data import DataLoader

from src.data import MultiTaskImageDataset, ImageTextPairDataset
from src.models import PerceiverEncoder, PatchDecoder, SimpleDecoder, Stage1Classifier
from .train_standalone_cls import StandaloneImageClassifier

# Ensure 'src' package is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def random_mask_images(x: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """
    입력 이미지 텐서의 일부 픽셀을 0으로 마스킹.
    x: (B, C, H, W)
    """
    if mask_ratio <= 0:
        return x
    b, c, h, w = x.shape
    mask = torch.rand(b, 1, h, w, device=x.device) > mask_ratio
    return x * mask


def collate_images(batch: List[dict]) -> dict:
    """Collate for Stage1 image-only datasets.
    - Pads images in a batch to the same H,W if needed and stacks to (B,C,H,W).
    - Returns `images` tensor and `cond_ids` (from `cond_id` or `type_id`).
    """
    # filter out None entries
    batch = [b for b in batch if b is not None]
    if not batch:
        return {"images": torch.empty(0), "cond_ids": torch.empty(0, dtype=torch.long)}

    imgs = [b["image"] for b in batch]
    # If all shapes equal, stack directly
    same_size = all(img.shape == imgs[0].shape for img in imgs)
    if same_size:
        images = torch.stack(imgs)
    else:
        # pad to max H,W
        h_max = max(img.shape[-2] for img in imgs)
        w_max = max(img.shape[-1] for img in imgs)
        padded = []
        for img in imgs:
            _, h, w = img.shape
            pad_h = h_max - h
            pad_w = w_max - w
            pad = (0, pad_w, 0, pad_h)
            padded.append(F.pad(img, pad, value=0.0))
        images = torch.stack(padded)

    # cond ids
    if "cond_id" in batch[0]:
        cond_ids = torch.tensor([b["cond_id"] for b in batch], dtype=torch.long)
    else:
        cond_ids = torch.tensor([b.get("type_id", 0) for b in batch], dtype=torch.long)

    return {"images": images, "cond_ids": cond_ids}

# -------------------------------------------------------------
# Stage 1 Metrics & Visualization Helpers (PSNR, SSIM, Unnormalize)
# -------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

@torch.no_grad()
def unnormalize(img: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization back to [0,1] range (clamped)."""
    return (img * IMAGENET_STD.to(img.device) + IMAGENET_MEAN.to(img.device)).clamp(0.0, 1.0)

@torch.no_grad()
def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0:
        return float('inf')
    return 10.0 * math.log10((max_val ** 2) / mse)

@torch.no_grad()
def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, C1: float, C2: float, window: torch.Tensor) -> torch.Tensor:
    # x,y: (B,1,H,W)
    mu_x = F.conv2d(x, window, padding=window.size(-1)//2, groups=1)
    mu_y = F.conv2d(y, window, padding=window.size(-1)//2, groups=1)
    sigma_x = F.conv2d(x * x, window, padding=window.size(-1)//2, groups=1) - mu_x * mu_x
    sigma_y = F.conv2d(y * y, window, padding=window.size(-1)//2, groups=1) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, window, padding=window.size(-1)//2, groups=1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean(dim=[1,2,3])  # (B)

@torch.no_grad()
def compute_ssim_batch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute average SSIM for batch of RGB images in [0,1]."""
    # Based on standard SSIM with Gaussian window (11x11, sigma=1.5)
    if x.shape != y.shape:
        raise ValueError(f"SSIM shape mismatch: {x.shape} vs {y.shape}")
    b, c, h, w = x.shape
    if h < 11 or w < 11:
        return 0.0
    # Build (1,1,11,11) Gaussian kernel once
    coords = torch.arange(11, device=x.device) - 5
    gauss_1d = torch.exp(-(coords**2)/(2*1.5*1.5))
    gauss_1d = gauss_1d / gauss_1d.sum()
    window = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    window = window.view(1, 1, 11, 11)
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    # Compute per-channel then average
    ssim_vals = []
    for ch in range(c):
        xv = x[:, ch:ch+1]
        yv = y[:, ch:ch+1]
        ssim_ch = _ssim_per_channel(xv, yv, C1, C2, window)
        ssim_vals.append(ssim_ch)
    ssim_all = torch.stack(ssim_vals, dim=1).mean(dim=1)  # (B)
    return float(ssim_all.mean().item())

def _build_stage1_dataloader(args) -> tuple[DataLoader, Dict[str, int]]:
    """
    Build a Stage 1 dataloader that can read from multiple sources and produce per-image items with cond_id.
    Returns dataloader and the global condition map (name->id).
    """
    sources = [s.strip().lower() for s in args.stage1_sources.split(",") if s.strip()]
    use_wqa = "weatherqa" in sources
    use_s2 = "chatearthnet" in sources and bool(args.chatearthnet_json_path) and bool(args.chatearthnet_image_root)
    use_era5 = "climateiqa" in sources and bool(args.climateiqa_json_path) and bool(args.climateiqa_image_root)
    if not (use_wqa or use_s2 or use_era5):
        raise ValueError("No valid stage1 sources specified. Use --stage1-sources (weatherqa,imagefolder,chatearthnet,climateiqa) and related paths.")

    pretrain_years = [int(y) for y in args.pretrain_years.split(",")] if args.pretrain_years else None
    weatherqa_types = [t for t in args.weatherqa_include_types.split(",") if t] if args.weatherqa_include_types else None
    ds = MultiTaskImageDataset(
        weatherqa_json_path=str(ROOT / args.weatherqa_json_path) if use_wqa else None,
        weatherqa_image_root=str(ROOT / args.weatherqa_image_dir) if use_wqa else None,
        weatherqa_years=pretrain_years if use_wqa else None,
        chatearthnet_json_path=args.chatearthnet_json_path if use_s2 else None,
        chatearthnet_image_root=args.chatearthnet_image_root if use_s2 else None,
        climateiqa_json_path=args.climateiqa_json_path if use_era5 else None,
        climateiqa_image_root=args.climateiqa_image_root if use_era5 else None,
        image_size=args.image_size,
        weatherqa_include_types=weatherqa_types
    )
    ds_preprocessed = ImageTextPairDataset(
        weatherqa_json_path=str(ROOT / args.weatherqa_json_path) if use_wqa else None,
        weatherqa_image_root=str(ROOT / args.weatherqa_image_dir) if use_wqa else None,
        weatherqa_years=pretrain_years if use_wqa else None,
        climateiqa_json_path=args.climateiqa_json_path if use_era5 else None,
        climateiqa_image_root=args.climateiqa_image_root if use_era5 else None,
        image_size=args.image_size,
    )
    ds_preprocessed.save_data_items("stage1_preprocessed_items.json")
    #ds_preprocessed.load_data_items("stage1_preprocessed_items.json")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_images)
    return dl, ds.type_map

def train_stage1_mae(args, device, wandb=None) -> Path:
    """Stage 1: WeatherQA 이미지로 비전 인코더 사전학습 (MAE 유사)"""
    logging.info("[Stage 1] MAE 사전학습 시작")
    # Build dataloader across sources and global condition map
    dl, cond_map = _build_stage1_dataloader(args)
    image_classifier: Optional[StandaloneImageClassifier] = None

    # 모드에 따라 'Standalone Classifier'만 추가 로드/동결
    if args.stage1_cond_source == "gt":
        logging.info("Using 'gt' (Ground Truth) for cond_ids. (Standard MTL)")
        # 'image_classifier'는 None으로 유지
    
    elif args.stage1_cond_source == "pred":
        logging.info("Using 'pred' (Predicted) for cond_ids. (Noisy Condition MTL)")
        image_classifier = StandaloneImageClassifier(
            num_classes=len(cond_map),
            backbone_name=args.stage1_standalone_backbone,
        ).to(device)
        
        # 'pred' (Ablation) 모드: 사전학습된 가중치 로드
        try:
            ckpt_path = args.stage1_standalone_ckpt
            if not Path(ckpt_path).exists():
                 raise FileNotFoundError(f"Standalone classifier checkpoint not found: {ckpt_path}")
            
            image_classifier.load_state_dict(torch.load(ckpt_path, map_location=device))
            logging.info(f"Standalone classifier weights loaded from {ckpt_path}")
            
        except Exception as e:
            logging.error(f"Standalone classifier 로드 실패: {e}")
            logging.error("먼저 'train_classifier.py'를 실행하여 모델을 훈련시켜야 합니다.")
            raise
        
        # 분류기 '동결' (학습 안 함)
        image_classifier.eval()
        image_classifier.requires_grad_(False) 
        # (옵티마이저 params_to_optimize에는 추가되지 않음)
    
    else:
        raise ValueError(f"Unknown stage1_cond_source: {args.stage1_cond_source}")

    # ResNet 기반 encoder 사용
    if args.encoder_mode == "resnet":
        from src.models import ResNetEncoder
        encoder = ResNetEncoder(
            pretrained=True,
            output_dim=args.latent_dim
        ).to(device)
    elif args.encoder_mode == "perceiver":
        encoder = PerceiverEncoder(
            patch_size=args.perceiver_patch_size,
            latent_dim=args.latent_dim,
            num_latents=args.perceiver_num_latents,
            num_blocks=args.perceiver_num_blocks,
            num_heads=args.perceiver_num_heads,
            moe_num_experts=args.moe_num_experts,
            num_conditions=len(cond_map),
        ).to(device)
    elif args.encoder_mode == "perceiver_patch_mae":
        from src.models import ConditionalPerceiverEncoder
        encoder = ConditionalPerceiverEncoder(
            num_types=len(cond_map),
            patch_size=args.perceiver_patch_size,
            latent_dim=args.latent_dim,
            num_latents=args.perceiver_num_latents,
            num_blocks=args.perceiver_num_blocks,
            num_heads=args.perceiver_num_heads,
            moe_num_experts=args.moe_num_experts
        ).to(device)

    # 디코더 선택: patch 모드면 모든 latent를 사용해 패치 재조합, pooled 모드면 전역 벡터 복원
    if args.decoder_mode == "patch":
        decoder = PatchDecoder(
            latent_dim=args.latent_dim,
            num_latents=args.perceiver_num_latents,
            patch_size=args.perceiver_patch_size,
            image_size=args.image_size,
            out_channels=3,
        ).to(device)
        encoder_return_all = True
    else:
        decoder = SimpleDecoder(latent_dim=args.latent_dim, image_size=args.image_size).to(device)
        encoder_return_all = False

    # Classification head for condition prediction (multi-task)
    classifier = Stage1Classifier(input_dim=args.latent_dim, num_classes=len(cond_map)).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(classifier.parameters())
    optim = torch.optim.AdamW(params, lr=args.stage1_lr, weight_decay=args.weight_decay)

    # Inverse map for per-source logging
    inv_cond_map = {v: k for k, v in cond_map.items()}

    for epoch in range(1, args.stage1_epochs + 1):
        cls_answer_cnt = 0
        cls_total_cnt = 0

        logging.info(f"Epoch {epoch}/{args.stage1_epochs}, {len(dl)} batches")
        encoder.train(); decoder.train(); classifier.train()
        running = 0.0
        running_recon = 0.0
        running_cls = 0.0
        total_correct = 0
        total_count = 0
        # Per-source MSE accumulation
        source_mse_sum = {cid: 0.0 for cid in inv_cond_map.keys()}
        source_mse_count = {cid: 0 for cid in inv_cond_map.keys()}
        # PSNR & SSIM accumulation (computed on unnormalized [0,1])
        unnorm_mse_sum = 0.0
        ssim_sum = 0.0
        ssim_batches = 0
        # Sample images for visualization
        sample_logged = False
        sample_pack = {}
        for batch in tqdm(dl):
            imgs = batch["images"].to(device)
            cond_ids_gt = batch["cond_ids"].to(device) # '정답' ID
            masked = random_mask_images(imgs, mask_ratio=args.mask_ratio)
            
            cond_ids_for_encoder: torch.Tensor

            # 1) 'pred' 모드: 분류기를 '추론' 모드로 사용
            if image_classifier is not None:
                with torch.no_grad(): # 그래디언트 계산 비활성화
                    logits_pred = image_classifier(imgs)
                    preds = logits_pred.argmax(dim=-1)
                
                # ★ Encoder는 '예측된' ID를 조건으로 받음
                cond_ids_for_encoder = preds 
            
            # 2) 'gt' 모드: Encoder에 '정답' ID를 조건으로 줌
            else: 
                cond_ids_for_encoder = cond_ids_gt
            cls_answer_cnt += (cond_ids_for_encoder == cond_ids_gt).sum().item()
            cls_total_cnt += cond_ids_gt.numel()

            if args.encoder_mode == "resnet":
                lat = encoder(masked)
            elif args.encoder_mode == "perceiver_patch_mae":
                lat, _, _ = encoder(masked, type_ids=cond_ids_for_encoder, keep_ratio=(1 - args.mask_ratio))
            else:
                lat = encoder(masked, return_all=encoder_return_all, cond_ids=cond_ids_for_encoder)
            recon = decoder(lat)

            # Masked autoencoding loss
            #loss_recon = F.mse_loss(recon, imgs)
            loss_recon = F.l1_loss(recon, imgs)

            # Classification loss on pooled latent
            pooled = lat.mean(dim=1) if lat.dim() == 3 else lat
            logits = classifier(pooled)
            loss_cls = F.cross_entropy(logits, cond_ids_gt)
            # accuracy
            preds = logits.argmax(dim=-1)
            total_correct += (preds == cond_ids_gt).sum().item()
            total_count += cond_ids_gt.numel()

            new_cls_loss_weight = args.cls_loss_weight
            if args.enable_cls_weight_control:
                # 3 epoch 이후부터 분류 손실 가중치 조정
                if epoch >= 3:
                    new_cls_loss_weight = args.cls_loss_weight + (args.cls_weight_control * (epoch - 3))
            loss = (1 - new_cls_loss_weight) * loss_recon + new_cls_loss_weight * loss_cls

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optim.step()

            running += loss.item()
            running_recon += loss_recon.item()
            running_cls += loss_cls.item()
            # Per-source MSE (normalized space)
            with torch.no_grad():
                unique_ids = cond_ids_gt.unique()
                for uid in unique_ids:
                    mask = cond_ids_gt == uid
                    if mask.any():
                        mse_uid = F.mse_loss(recon[mask], imgs[mask], reduction='mean').item()
                        source_mse_sum[int(uid.item())] += mse_uid * mask.sum().item()
                        source_mse_count[int(uid.item())] += mask.sum().item()
                # PSNR/SSIM on unnormalized images
                if args.stage1_enable_metrics:
                    orig_unn = unnormalize(imgs)
                    recon_unn = unnormalize(recon)
                    batch_mse_unn = F.mse_loss(recon_unn, orig_unn, reduction='mean').item()
                    unnorm_mse_sum += batch_mse_unn
                    try:
                        ssim_val = compute_ssim_batch(recon_unn, orig_unn)
                        ssim_sum += ssim_val
                        ssim_batches += 1
                    except Exception as _e:
                        pass
                # Capture samples from first batch for visualization
                if (not sample_logged) and wandb is not None and args.stage1_log_images > 0 and (epoch % max(1, args.stage1_image_log_interval) == 0):
                    K = min(args.stage1_log_images, imgs.size(0))
                    sample_pack = {
                        'orig': orig_unn[:K].detach().cpu(),
                        'masked': unnormalize(masked)[:K].detach().cpu(),
                        'recon': recon_unn[:K].detach().cpu(),
                        'cond_ids': cond_ids_gt[:K].detach().cpu(),
                    }
                    sample_logged = True
        avg_total = running / max(1, len(dl))
        avg_recon = running_recon / max(1, len(dl))
        avg_cls = running_cls / max(1, len(dl))
        acc = (total_correct / max(1, total_count)) if total_count > 0 else 0.0
        # Per-source averaged MSE
        per_source_mse = {}
        for cid, s in source_mse_sum.items():
            cnt = source_mse_count[cid]
            if cnt > 0:
                per_source_mse[inv_cond_map[cid]] = s / cnt
        # PSNR & SSIM
        if args.stage1_enable_metrics and len(dl) > 0:
            avg_unnorm_mse = unnorm_mse_sum / len(dl)
            psnr_val = compute_psnr(avg_unnorm_mse, max_val=1.0)
            ssim_val_epoch = (ssim_sum / max(1, ssim_batches)) if ssim_batches > 0 else 0.0
        else:
            psnr_val = None
            ssim_val_epoch = None
        logging.info(f"[Stage1][Epoch {epoch}] total: {avg_total:.4f} | recon: {avg_recon:.4f} | cls: {avg_cls:.4f} | acc: {acc:.4f}")
        if args.stage1_enable_metrics and psnr_val is not None:
            logging.info(f"[Stage1][Epoch {epoch}] PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val_epoch:.4f}")
        if per_source_mse:
            logging.info("[Stage1] Per-source MSE: " + ", ".join([f"{k}:{v:.4f}" for k,v in per_source_mse.items()]))
        logging.info(f"[Stage1] Condition prediction accuracy this epoch: {cls_answer_cnt}/{cls_total_cnt} = {cls_answer_cnt/ max(1,cls_total_cnt):.4f}")
        if wandb is not None:
            wandb.log({
                "stage": 1,
                "epoch": epoch,
                "stage1/total_loss": avg_total,
                "stage1/recon_loss": avg_recon,
                "stage1/cls_loss": avg_cls,
                "stage1/cls_acc": acc,
                **({"stage1/psnr": psnr_val} if psnr_val is not None else {}),
                **({"stage1/ssim": ssim_val_epoch} if ssim_val_epoch is not None else {}),
            })
            # Log per-source MSE
            for src_name, mse_val in per_source_mse.items():
                wandb.log({f"stage1/mse/{src_name}": mse_val, "epoch": epoch})
            # Log reconstruction samples
            if sample_pack and sample_logged:
                images_to_log = []
                for i in range(sample_pack['orig'].size(0)):
                    cid = int(sample_pack['cond_ids'][i].item())
                    cname = inv_cond_map.get(cid, str(cid))
                    panel = torch.cat([
                        sample_pack['orig'][i],
                        sample_pack['masked'][i],
                        sample_pack['recon'][i],
                    ], dim=2)  # concatenate horizontally (C,H,3W)
                    images_to_log.append(wandb.Image(panel, caption=f"{cname}"))
                wandb.log({"stage1/recon_samples": images_to_log, "epoch": epoch})

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "stage1_vision_encoder_mae.pt"
    torch.save(encoder.state_dict(), ckpt)
    logging.info(f"[Stage 1] 저장 완료: {ckpt}")
    return ckpt


# -------------------------------------------------------------
# Stage 2: LLM 임베딩과의 정렬(Alignment) - LLM은 고정, 인코더+프로젝터 학습
# -------------------------------------------------------------
def train_stage2_alignment(args, device, encoder_ckpt: Optional[Path], wandb=None) -> tuple[Path, Path, dict]:
    """Stage 2: WeatherQA 질문 텍스트와의 특징 정렬. LLM은 고정, 인코더+프로젝터 학습"""
    logging.info("TODO: text-image alignment 코드 구현 통합 필요")
    return None, None, {}


# -------------------------------------------------------------
# Stage 3: LoRA로 LLM 미세조정 + 비전 인코더/프로젝터 지속 업데이트
# -------------------------------------------------------------
def train_stage3_vlm(args, device, encoder_ckpt: Path, projector_ckpt: Path, type_map: dict, wandb=None):
    logging.info("TODO: VLM fine-tuning 코드 구현 통합 필요")
    return None
    

def parse_args():
    p = argparse.ArgumentParser(description="Three-stage VLM training (WeatherQA-first)")
    # 공통
    p.add_argument("--output-dir", type=str, default="checkpoints/three_stage")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--image-size", type=int, default=224)

    # 데이터 (WeatherQA)
    p.add_argument("--weatherqa-json-path", type=str, default="data/WeatherQA/dataset_2014-2020.json")
    p.add_argument("--weatherqa-image-dir", type=str, default="data/WeatherQA/WeatherQA_MD_2014-2019")
    p.add_argument("--weatherqa-include-types", type=str, default="", help="Comma-separated WeatherQA condition types to include (default: all)")

    # Perceiver 설정
    p.add_argument("--latent-dim", type=int, default=768)
    p.add_argument("--perceiver-patch-size", type=int, default=16)
    p.add_argument("--perceiver-num-latents", type=int, default=256)
    p.add_argument("--perceiver-num-blocks", type=int, default=6)
    p.add_argument("--perceiver-num-heads", type=int, default=8)
    p.add_argument("--moe-num-experts", type=int, default=8)

    # Stage 1 (MAE)
    p.add_argument("--stage1-epochs", type=int, default=5)
    p.add_argument("--stage1-lr", type=float, default=1e-4)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--pretrain-years", type=str, default="2019,2020,2021,2022,2023,2024")
    p.add_argument("--encoder-mode", type=str, default="perceiver", choices=["perceiver", "perceiver_patch_mae", "resnet"], help="Stage1 인코더 유형")
    p.add_argument("--decoder-mode", type=str, default="pooled", choices=["pooled", "patch"], help="Stage1 디코더 유형: pooled는 전역 벡터 복원, patch는 모든 latent로 패치 재조합")
    p.add_argument("--stage1-sources", type=str, default="weatherqa", help="Comma-separated sources: weatherqa,imagefolder,chatearthnet,climateiqa")
    p.add_argument("--imagefolder-root", type=str, default="", help="Root path for generic imagefolder source")
    p.add_argument("--chatearthnet-json-path", type=str, default="", help="ChatEarthNet metadata JSON path")
    p.add_argument("--chatearthnet-image-root", type=str, default="", help="ChatEarthNet image root containing band folders")
    p.add_argument("--climateiqa-json-path", type=str, default="", help="ClimateIQA metadata JSON path")
    p.add_argument("--climateiqa-image-root", type=str, default="", help="ClimateIQA tensor root")
    p.add_argument("--cls-loss-weight", type=float, default=0.2, help="Weight for condition classification loss in Stage 1")

    # Standalone classifier를 사용할 경우, 사용할 백본과 체크포인트 경로
    p.add_argument("--stage1-cond-source", type=str, default="gt", choices=["gt", "pred"],
                     help="Source of cond_ids for Encoder. 'gt' uses Ground Truth (MTL), 'pred' uses a pre-trained standalone classifier.")
    p.add_argument("--stage1-standalone-backbone", type=str, default="efficientnet_b0",
                     help="Backbone for the standalone classifier (if stage1_cond_source='pred')")
    p.add_argument("--stage1-standalone-ckpt", type=str, default="/home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt",
                     help="Path to the pre-trained standalone classifier checkpoint (if stage1_cond_source='pred')")

    p.add_argument("--enable-cls-weight-control", action="store_true", help="Enable dynamic weight control for classification loss")
    p.add_argument("--cls-weight-control", type=float, default=0.05, help="Weight control for preventing exploration of classification loss")

    # Stage 1 metrics & visualization
    p.add_argument("--stage1-enable-metrics", action="store_true", help="Compute and log PSNR/SSIM & per-source MSE for Stage1")
    p.add_argument("--stage1-log-images", type=int, default=4, help="Number of reconstruction sample triplets (orig/masked/recon) to log per epoch (0 = disable)")
    p.add_argument("--stage1-image-log-interval", type=int, default=1, help="Epoch interval for logging reconstruction samples")

    # Stage 2 (Alignment)
    p.add_argument("--stage2-epochs", type=int, default=3)
    p.add_argument("--stage2-lr", type=float, default=5e-5)
    p.add_argument("--embedding-model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--llm-model-name", type=str, default="google/gemma-2b-it")
    p.add_argument("--proj-hidden-dim", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--align-years", type=str, default="2019,2020,2021,2022,2023,2024")
    p.add_argument("--stage1-encoder-ckpt", type=str, default="", help="Stage 2 시작 시 사용할 Stage 1 encoder 체크포인트 경로")

    # Stage 3 (VLM fine-tuning with LoRA)
    p.add_argument("--stage3-epochs", type=int, default=3)
    p.add_argument("--stage3-lr", type=float, default=5e-5)
    p.add_argument("--finetune-years", type=str, default="2020")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # 실행 제어
    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--only-stage3", action="store_true", help="Stage2 산출물(encoder/projector) 경로를 추가로 지정해야 함")
    p.add_argument("--stage2-encoder-ckpt", type=str, default="")
    p.add_argument("--stage2-projector-ckpt", type=str, default="")

    # wandb
    p.add_argument("--wandb-enable", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="ClimateToText", help="wandb project name")
    p.add_argument("--wandb-entity", type=str, default="", help="wandb entity (optional)")
    p.add_argument("--wandb-run-name", type=str, default="", help="wandb run name (optional)")

    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # wandb init (optional)
    wandb = None
    if args.wandb_enable:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project=args.wandb_project,
                entity=(args.wandb_entity or None),
                name=(args.wandb_run_name or None),
                config=vars(args),
            )
        except Exception as e:
            logging.warning(f"wandb init 실패: {e}. wandb 비활성화합니다.")
            wandb = None

    ROOT.joinpath(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(ROOT / args.output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 순차 실행 (필요 시 스킵 가능)
    stage1_ckpt = None
    stage2_encoder_ckpt = Path(args.stage2_encoder_ckpt) if args.stage2_encoder_ckpt else None
    stage2_projector_ckpt = Path(args.stage2_projector_ckpt) if args.stage2_projector_ckpt else None

    if not args.only_stage3 and not args.skip_stage1:
        stage1_ckpt = train_stage1_mae(args, device, wandb)
    elif args.skip_stage1 and args.stage1_encoder_ckpt:
        # 외부 Stage1 결과를 사용하여 Stage2부터 시작
        stage1_ckpt = Path(args.stage1_encoder_ckpt)
        if not stage1_ckpt.exists():
            raise FileNotFoundError(f"지정한 Stage1 체크포인트를 찾을 수 없습니다: {stage1_ckpt}")

    if not args.only_stage3 and not args.skip_stage2:
        stage2_encoder_ckpt, stage2_projector_ckpt, type_map = train_stage2_alignment(args, device, stage1_ckpt, wandb)
    else:
        # 외부에서 전달된 경로 사용
        type_map = {}
        if (not stage2_encoder_ckpt) or (not stage2_projector_ckpt):
            raise ValueError("--only-stage3 또는 stage2 스킵을 사용할 경우, --stage2-encoder-ckpt 및 --stage2-projector-ckpt를 제공해야 합니다.")

    # Stage 3 실행
    train_stage3_vlm(args, device, stage2_encoder_ckpt, stage2_projector_ckpt, type_map, wandb)


## Main function ##########################################################################
if __name__ == "__main__":
    main()