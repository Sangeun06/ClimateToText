"""
train_classifier.py

Stage 1의 'Ablation Study'를 위한 'Standalone' 이미지 분류기 단독 학습 스크립트.
ImageNet으로 사전학습된 EfficientNet/ResNet을 사용하여, 원본 이미지만으로
cond_id (데이터셋 종류)를 분류하도록 학습합니다.

- train_stage1_mae.py와 동일한 데이터로더(_build_stage1_dataloader)를 사용합니다.
- Validation Set을 기준으로 가장 좋은 모델을 저장합니다.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from .helpers import setup_logging, set_seed, collate_images

# src.data에서 데이터셋 클래스 임포트
try:
    from src.data import MultiTaskImageDataset
except ImportError:
    print("Error: src.data.MultiTaskImageDataset를 찾을 수 없습니다.")
    sys.exit(1)


# --- StandaloneImageClassifier 정의 (EfficientNet/ResNet) ---

class StandaloneImageClassifier(nn.Module):
    """
    원본 이미지를 직접 입력받아 cond_id를 분류하는 '똑똑한' 분류기.
    ImageNet으로 사전학습된 백본을 사용합니다.
    """
    def __init__(self, num_classes: int, backbone_name: str = "efficientnet_b0"):
        super().__init__()
        
        self.backbone_name = backbone_name
        if self.backbone_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            num_features = self.backbone.classifier[1].in_features # (1280)
            # 분류기 헤드 교체
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        
        elif self.backbone_name == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
            num_features = self.backbone.fc.in_features # (512)
            # 분류기 헤드 교체
            self.backbone.fc = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"지원되지 않는 백본: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# --- 데이터로더 빌더 (train_stage1_mae.py와 동일 로직) ---

def _build_dataloader(args, years: List[int]) -> tuple[Optional[DataLoader], Dict[str, int]]:
    """지정된 연도로 Stage 1 데이터로더를 빌드합니다."""
    
    # 이 부분은 train_stage1_mae.py의 _build_stage1_dataloader와
    # 동일한 로직을 사용해야 합니다.
    # (간결화를 위해 MultiTaskImageDataset만 사용하도록 단순화)
    
    ds = MultiTaskImageDataset(
        weatherqa_json_path=str(args.weatherqa_json_path) if args.weatherqa_json_path else None,
        weatherqa_image_root=str(args.weatherqa_image_dir) if args.weatherqa_image_dir else None,
        weatherqa_years=years,
        
        chatearthnet_json_path=args.chatearthnet_json_path if args.chatearthnet_json_path else None,
        chatearthnet_image_root=args.chatearthnet_image_root if args.chatearthnet_image_root else None,
        
        climateiqa_json_path=args.climateiqa_json_path if args.climateiqa_json_path else None,
        climateiqa_image_root=args.climateiqa_image_root if args.climateiqa_image_root else None,
        
        image_size=args.image_size, # (중요) 224x224로 리사이즈
    )

    if len(ds) == 0:
        logging.warning(f"{years} 연도에 해당하는 데이터가 없습니다.")
        return None, {}

    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        collate_fn=collate_images
    )
    return dl, ds.type_map


# --- 메인 학습/검증 로직 ---

def main(args):
    setup_logging()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. 데이터 로드 (Train / Validation 분리)
    train_years = [int(y) for y in args.train_years.split(",")]
    val_years = [int(y) for y in args.val_years.split(",")]
    
    logging.info(f"Loading train data (Years: {train_years})...")
    train_dl, cond_map = _build_dataloader(args, train_years)
    logging.info(f"Loading validation data (Years: {val_years})...")
    val_dl, _ = _build_dataloader(args, val_years)

    if train_dl is None or val_dl is None:
        logging.error("학습 또는 검증 데이터로더를 생성할 수 없습니다. 경로와 연도를 확인하세요.")
        return

    num_classes = len(cond_map)
    logging.info(f"Classifier MClasses: {num_classes} ({cond_map})")

    # 2. 모델 및 옵티마이저 초기화
    model = StandaloneImageClassifier(
        num_classes=num_classes,
        backbone_name=args.backbone
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 3. 학습 및 검증 루프
    best_val_acc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / f"standalone_classifier_{args.backbone}_best.pt"

    for epoch in range(1, args.epochs + 1):
        logging.info(f"--- Epoch {epoch}/{args.epochs} ---")
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_dl, desc="Training"):
            imgs = batch["images"].to(device)
            cond_ids_gt = batch["cond_ids"].to(device)

            logits = model(imgs)
            loss = F.cross_entropy(logits, cond_ids_gt)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == cond_ids_gt).sum().item()
            train_total += cond_ids_gt.numel()
        
        avg_train_loss = train_loss / len(train_dl)
        avg_train_acc = (train_correct / train_total) * 100
        logging.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Validation"):
                imgs = batch["images"].to(device)
                cond_ids_gt = batch["cond_ids"].to(device)
                
                logits = model(imgs)
                loss = F.cross_entropy(logits, cond_ids_gt)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == cond_ids_gt).sum().item()
                val_total += cond_ids_gt.numel()

        avg_val_loss = val_loss / len(val_dl)
        avg_val_acc = (val_correct / val_total) * 100
        logging.info(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")

        # --- Save Best Model ---
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"★★★ New Best Model Saved (Acc: {best_val_acc:.2f}%) to {best_model_path} ★★★")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Standalone Image Classifier Training Script")
    
    # 공통 경로 (train_stage1_mae.py와 동일하게)
    p.add_argument("--image-size", type=int, default=224)
    # p.add_argument("--weatherqa-json-path", type=str, default="data/WeatherQA/dataset_2014-2020.json")
    # p.add_argument("--weatherqa-image-dir", type=str, default="data/WeatherQA/WeatherQA_MD_2014-2019")
    p.add_argument("--weatherqa-json-path", type=str, default="/home/agi592/kse/ClimateToText/data/WeatherQA/dataset_2014-2020.json")
    p.add_argument("--weatherqa-image-dir", type=str, default="/home/agi592/kse/ClimateToText/data/WeatherQA/WeatherQA_MD_2014-2019")
    p.add_argument("--chatearthnet-json-path", type=str, default="")
    p.add_argument("--chatearthnet-image-root", type=str, default="")
    p.add_argument("--climateiqa-json-path", type=str, default="")
    p.add_argument("--climateiqa-image-root", type=str, default="")

    # 학습 파라미터
    p.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    
    # (중요) Train/Val 연도 분리
    p.add_argument("--train-years", type=str, default="2014,2015,2016,2017")
    p.add_argument("--val-years", type=str, default="2018")
    
    p.add_argument("--output-dir", type=str, default="checkpoints/standalone_classifier")
    
    args = p.parse_args()
    main(args)