import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# (기존 프로젝트 모듈 임포트)
from src.models import PerceiverEncoder
from script.train_image_encoder import _build_stage1_dataloader, setup_logging, set_seed

def extract_features(loader, model, device, num_samples=2000, encoder_mode: str = "perceiver"):
    """
    데이터로더에서 일정 개수(num_samples)만큼 데이터를 뽑아
    Latent Vector와 정답 라벨(Label)을 추출합니다.
    """
    model.eval()
    all_latents = []
    all_labels = []
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Features"):
            imgs = batch["images"].to(device)
            cond_ids = batch["cond_ids"].to(device)

            # 1. Encoder 실행 (Latent 추출)
            #    인코더 타입에 따라 호출 방식이 다름
            if encoder_mode == "perceiver":
                # PerceiverEncoder: (B,C,H,W) + cond_ids, return_all=False -> (B,D)
                lat = model(imgs, return_all=False, cond_ids=cond_ids)
            elif encoder_mode == "perceiver_patch_mae":
                # ConditionalPerceiverEncoder: MAE용, (이미지, type_ids, keep_ratio) -> (B,L,D), ...
                # t-SNE에서는 글로벌 벡터가 필요하므로 mean pooling
                lat_seq, _, _ = model(imgs, type_ids=cond_ids, keep_ratio=1.0)
                lat = lat_seq.mean(dim=1)  # (B,D)
            elif encoder_mode == "resnet":
                # ResNetEncoder: (B,C,H,W) -> (B,L,D), 여기서도 mean pooling
                lat_seq = model(imgs)
                lat = lat_seq.mean(dim=1)  # (B,D)
            else:
                # 예비: 정의되지 않은 모드는 그냥 출력 사용
                lat = model(imgs)
            
            all_latents.append(lat.cpu().numpy())
            all_labels.append(cond_ids.cpu().numpy())
            
            count += imgs.size(0)
            if count >= num_samples:
                break
                
    # 리스트를 하나의 거대한 numpy 배열로 합침
    all_latents = np.concatenate(all_latents, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]
    
    return all_latents, all_labels

def plot_tsne(latents, labels, inv_cond_map, save_path, title="t-SNE of Latents"):
    """
    t-SNE를 수행하고 시각화합니다.
    """
    print("Running t-SNE... (이 작업은 몇 분 걸릴 수 있습니다)")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    latents_2d = tsne.fit_transform(latents)
    
    print("Plotting...")
    plt.figure(figsize=(14, 10)) # figsize를 약간 더 키워서 범례 공간 확보
    
    # 라벨 숫자를 실제 이름으로 변환
    label_names = [inv_cond_map[l] for l in labels]
    
    # Seaborn으로 산점도 그리기
    sns.scatterplot(
        x=latents_2d[:, 0], 
        y=latents_2d[:, 1], 
        hue=label_names, 
        # ★★★ 이 부분을 수정합니다 ★★★
        # palette="tab10", # 기존: 10개 색상만 제공, 반복
        palette="tab20",  # 새로운 팔레트: 20개 색상 제공
        # palette="hls",   # 대안1: Hue, Lightness, Saturation 기반, 더 많은 색상
        # palette="husl",  # 대안2: hls와 유사하지만 좀 더 균일한 색상
        # palette=sns.color_palette("Spectral", n_colors=len(inv_cond_map)), # 동적으로 n_colors 지정
        # ★★★ 수정 완료 ★★★
        s=60,            # 점 크기
        alpha=0.7        # 투명도
    )
    
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    
    # 범례 위치 조정 (그래프 밖으로 빼서 겹치지 않게)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.) 
    
    plt.tight_layout() # 그래프와 범례가 잘 맞도록 자동 조정
    
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE 그래프 저장 완료: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="학습된 인코더 체크포인트 경로 (.pt)")
    # (데이터 로더 생성을 위한 필수 인자들 - train_pipeline.py의 기본값 사용 추천)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--weatherqa-json-path", type=str, default="/home/agi592/kse/ClimateToText/data/WeatherQA/dataset_2014-2020.json")
    parser.add_argument("--weatherqa-image-dir", type=str, default="/home/agi592/kse/ClimateToText/data/WeatherQA/WeatherQA_MD_2014-2019")
    parser.add_argument("--weatherqa-include-types", type=str, default="", help="Comma-separated WeatherQA condition types to include (default: all)")
    parser.add_argument("--encoder-mode", type=str, default="perceiver", choices=["perceiver", "perceiver_patch_mae", "resnet"], help="Stage1 인코더 유형")
    # ... 필요한 경우 다른 경로 인자도 추가 ...
    
    # 모델 설정 (학습 때와 동일해야 함)
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--perceiver-num-latents", type=int, default=256)
    parser.add_argument("--perceiver-patch-size", type=int, default=16)
    # ...
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로더 빌드 (Validation 셋 사용 추천)
    #    (편의상 train_pipeline의 함수 재사용)
    #    *주의: 시각화용이므로 batch_size는 적당히 32 정도
    args.batch_size = 32
    args.num_workers = 4
    # args에 없는 필드들 채워주기 (dataloader 함수가 요구하는 것들)
    args.stage1_sources = "weatherqa" # (예시)
    args.pretrain_years = "2019"      # (예시) 검증용 연도
    args.chatearthnet_json_path = ""
    args.climateiqa_json_path = ""
    
    print("Loading Data...")
    # _build_stage1_dataloader는 args 객체를 받으므로 args에 필요한 속성이 다 있어야 함
    # (실제 사용 시에는 train_pipeline.py의 parse_args()를 활용하거나 필요한 필드를 args에 수동 추가)
    dl, cond_map = _build_stage1_dataloader(args)
    inv_cond_map = {v: k for k, v in cond_map.items()}
    
    # 2. 모델 로드
    print(f"Loading Model from {args.ckpt}...")
    encoder = PerceiverEncoder(
        patch_size=args.perceiver_patch_size,  # 학습 설정과 동일하게
        latent_dim=args.latent_dim,
        num_latents=args.perceiver_num_latents,
        num_blocks=6,          # 학습 설정과 동일하게
        num_heads=8,           # 학습 설정과 동일하게
        moe_num_experts=8,     # 학습 설정과 동일하게
        num_conditions=len(cond_map)
    ).to(device)
    
    # ResNet 기반 encoder 사용
    if args.encoder_mode == "resnet":
        from src.models import ResNetEncoder
        encoder = ResNetEncoder(
            pretrained=True,
            output_dim=args.latent_dim
        ).to(device)
    elif args.encoder_mode == "perceiver":
        encoder = PerceiverEncoder(
            patch_size=args.perceiver_patch_size,  # 학습 설정과 동일하게
            latent_dim=args.latent_dim,
            num_latents=args.perceiver_num_latents,
            num_blocks=6,          # 학습 설정과 동일하게
            num_heads=8,           # 학습 설정과 동일하게
            moe_num_experts=8,     # 학습 설정과 동일하게
            num_conditions=len(cond_map),
        ).to(device)
    elif args.encoder_mode == "perceiver_patch_mae":
        from src.models import ConditionalPerceiverEncoder
        encoder = ConditionalPerceiverEncoder(
            num_types=len(cond_map),
            patch_size=args.perceiver_patch_size,
            image_size=args.image_size,
            latent_dim=args.latent_dim,
            num_latents=args.perceiver_num_latents,
            num_blocks=6,
            num_heads=8,
            moe_num_experts=8
        ).to(device)
    
    encoder.load_state_dict(torch.load(args.ckpt, map_location=device))
    
    # 3. 특징 추출
    print("Extracting Features...")
    latents, labels = extract_features(dl, encoder, device, num_samples=2000, encoder_mode=args.encoder_mode)
    
    # 4. t-SNE 및 시각화
    # title = f"Latent Space t-SNE ({Path(args.ckpt).stem})"
    title = f"Latent Space t-SNE ({Path(args.ckpt).parent.name})" # 파일의 부모 디렉터리 이름
    
    save_path = f"tsne_result_tab20_{Path(args.ckpt).parent.name}.png" # 파일명 변경 (옵션)
    
    plot_tsne(latents, labels, inv_cond_map, save_path, title)

if __name__ == "__main__":
    main()
