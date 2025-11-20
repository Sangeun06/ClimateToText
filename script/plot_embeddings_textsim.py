import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

# src 디렉터리를 import path에 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (기존 프로젝트 모듈 임포트)
from src.data import get_default_image_transform, pil_loader
from src.models import PerceiverEncoder

# 동일 repo 내 텍스트 임베딩 유사도 클래스 사용
from .text_similarity_embedding import EmbeddingSimilarityScorer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def build_cond_map_from_json(json_path: str) -> Dict[str, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    cond_names = sorted({it.get("cond_name") for it in items if it.get("cond_name")})
    return {name: i for i, name in enumerate(cond_names)}


def select_items_for_tsne(
    json_path: str,
    cond_map: Dict[str, int],
    num_samples: int,
    seed: int,
    target_cond_name: str = "",
) -> Tuple[List[Dict], List[int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # cond_map에 존재하고, 이미지 파일이 실제로 있는 것만 사용
    filtered: List[Dict] = []
    indices: List[int] = []
    for idx, it in enumerate(items):
        img_path = it.get("image")
        cond_name = it.get("cond_name")
        if not img_path or not cond_name:
            continue
        if cond_name not in cond_map:
            continue
        if target_cond_name and cond_name != target_cond_name:
            continue
        if not os.path.exists(img_path):
            logging.warning(f"이미지 파일 없음, 스킵: {img_path}")
            continue
        filtered.append(it)
        indices.append(idx)

    if not filtered:
        raise RuntimeError(f"유효한 샘플을 찾지 못했습니다: {json_path}")

    rng = np.random.default_rng(seed)
    if num_samples > 0 and len(filtered) > num_samples:
        chosen_idx = rng.choice(len(filtered), size=num_samples, replace=False)
        selected = [filtered[i] for i in chosen_idx]
        selected_indices = [indices[i] for i in chosen_idx]
    else:
        selected = filtered
        selected_indices = indices

    return selected, selected_indices


def extract_features_from_items(
    items: List[Dict],
    cond_map: Dict[str, int],
    model: torch.nn.Module,
    device: torch.device,
    encoder_mode: str = "perceiver",
    image_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    JSON에서 고른 (image, annotation, cond_name) 샘플들에 대해
    이미지 인코더 latent와 cond_id, 텍스트를 추출.
    """
    model.eval()
    transform = get_default_image_transform(size=image_size)

    latents: List[np.ndarray] = []
    labels: List[int] = []
    texts: List[str] = []

    for it in tqdm(items, desc="Extracting Features"):
        img_path = it["image"]
        cond_name = it["cond_name"]
        text = it.get("annotation", "")

        img = pil_loader(Path(img_path))
        img_tensor = transform(img).unsqueeze(0).to(device)  # (1,C,H,W)
        cond_id = cond_map[cond_name]
        cond_ids = torch.tensor([cond_id], dtype=torch.long, device=device)

        with torch.no_grad():
            if encoder_mode == "perceiver":
                lat = model(img_tensor, return_all=False, cond_ids=cond_ids)  # (1,D)
            elif encoder_mode == "perceiver_patch_mae":
                lat_seq, _, _ = model(
                    img_tensor, type_ids=cond_ids, keep_ratio=1.0
                )  # (1,L,D)
                lat = lat_seq.mean(dim=1)  # (1,D)
            elif encoder_mode == "resnet":
                lat_seq = model(img_tensor)  # (1,L,D) 또는 (1,D)
                if lat_seq.dim() == 3:
                    lat = lat_seq.mean(dim=1)
                else:
                    lat = lat_seq
            else:
                lat = model(img_tensor)

        latents.append(lat.cpu().numpy()[0])
        labels.append(cond_id)
        texts.append(text)

    latents_arr = np.stack(latents, axis=0)
    labels_arr = np.array(labels, dtype=np.int64)
    return latents_arr, labels_arr, texts


def run_tsne(latents: np.ndarray, seed: int = 42) -> np.ndarray:
    print("Running t-SNE... (이 작업은 몇 분 걸릴 수 있습니다)")
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
    return tsne.fit_transform(latents)


def plot_tsne(
    latents_2d: np.ndarray,
    labels: np.ndarray,
    inv_cond_map: Dict[int, str],
    save_path: str,
    title: str = "t-SNE of Latents",
) -> None:
    print("Plotting (label-based)...")
    plt.figure(figsize=(14, 10))

    label_names = [inv_cond_map[int(l)] for l in labels]

    sns.scatterplot(
        x=latents_2d[:, 0],
        y=latents_2d[:, 1],
        hue=label_names,
        palette="tab20",
        s=60,
        alpha=0.7,
    )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE (label) 그래프 저장 완료: {save_path}")
    plt.show()


def compute_text_similarities(
    texts: List[str],
    ref_index: int,
    model_name_or_path: str,
    device: torch.device,
) -> np.ndarray:
    scorer = EmbeddingSimilarityScorer(
        model_name_or_path=model_name_or_path, 
        device='cpu', 
        # device=str(device), 
        normalize=True
    )
    embs = scorer._encode(texts)  # (N,D), L2-normalized
    ref_emb = embs[ref_index]
    sims = (embs @ ref_emb).cpu().numpy()  # cosine similarity

    # 0~1로 정규화 (컬러맵용)
    sims_min = float(np.min(sims))
    sims_max = float(np.max(sims))
    if sims_max - sims_min < 1e-8:
        return np.zeros_like(sims)
    return (sims - sims_min) / (sims_max - sims_min)


def plot_tsne_with_similarity(
    latents_2d: np.ndarray,
    sim_scores: np.ndarray,
    save_path: str,
    title: str,
    ref_index: int,
) -> None:
    print("Plotting (text-similarity-based)...")
    plt.figure(figsize=(14, 10))

    sc = plt.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=sim_scores,
        cmap="viridis",
        s=60,
        alpha=0.8,
    )
    plt.colorbar(sc, label="Text similarity to reference")

    # 기준 샘플은 강조해서 표시
    plt.scatter(
        latents_2d[ref_index, 0],
        latents_2d[ref_index, 1],
        c="red",
        edgecolors="black",
        s=100,
        label="Reference sample",
    )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE (text similarity) 그래프 저장 완료: {save_path}")
    plt.show()


def extract_image_id_from_path(image_path: str) -> str:
    """
    예시 경로:
      /.../md1636_20150807_23_ttd.gif  ->  md1636
    """
    base = os.path.basename(image_path)
    stem, _ = os.path.splitext(base)
    return stem.split("_")[0] if "_" in stem else stem


def save_tsne_html_with_similarity(
    latents_2d: np.ndarray,
    sim_scores: np.ndarray,
    ids: List[str],
    annotations: List[str],
    image_paths: List[str],
    ref_index: int,
    html_path: str,
    title: str,
    base_url: str = "",
) -> None:
    """
    Plotly CDN을 사용하는 간단한 HTML 인터랙티브 플롯.
    - 마우스 오버 시 ID + 텍스트(annotation) + 이미지가 표시됨.
    - 색깔은 sim_scores (0~1) 기준으로 그라데이션.
    """
    x = latents_2d[:, 0].tolist()
    y = latents_2d[:, 1].tolist()
    color = sim_scores.tolist()

    # 이미지 URL 구성: base_url이 있으면 http://host:port + 로컬 경로,
    # 없으면 file:// 스킴으로 직접 파일 접근
    if base_url:
        base = base_url.rstrip("/")
        image_urls = [f"{base}{p}" if p else "" for p in image_paths]
    else:
        image_urls = [
            f"file://{p}" if p and not p.startswith("file://") else p
            for p in image_paths
        ]

    data_dict = {
        "x": x,
        "y": y,
        "color": color,          # 0~1 정규화된 유사도 (색상용)
        "sims": color,           # 같은 값을 툴팁 숫자 표기를 위해 재사용
        "ids": ids,
        "annotations": annotations,
        "images": image_urls,
        "ref_index": int(ref_index),
        "title": title,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
</head>
<body>
  <div id="image-tooltip" style="position: fixed; display: none; pointer-events: none;
       border: 1px solid #ccc; background-color: white; padding: 4px; z-index: 9999;">
  </div>
  <div id="tsne-plot" style="width: 100%; height: 100vh;"></div>
  <script>
    const data = {json.dumps(data_dict, ensure_ascii=False)};

    const trace = {{
      x: data.x,
      y: data.y,
      mode: "markers",
      type: "scattergl",
      marker: {{
        color: data.color,
        colorscale: "Viridis",
        showscale: true,
        size: 8,
        opacity: 0.8
      }},
      name: "Samples"
    }};

    const refTrace = {{
      x: [data.x[data.ref_index]],
      y: [data.y[data.ref_index]],
      mode: "markers",
      type: "scattergl",
      marker: {{
        color: "red",
        size: 12,
        line: {{ color: "black", width: 1 }}
      }},
      name: "Reference"
    }};

    const layout = {{
      title: data.title,
      xaxis: {{ title: "t-SNE Dim 1" }},
      yaxis: {{ title: "t-SNE Dim 2" }},
      hovermode: "closest"
    }};

    Plotly.newPlot("tsne-plot", [trace, refTrace], layout);

    const plotDiv = document.getElementById("tsne-plot");
    const tooltip = document.getElementById("image-tooltip");

    plotDiv.on("plotly_hover", function(evt) {{
      if (!evt || !evt.points || evt.points.length === 0) return;
      const pt = evt.points[0];
      const idx = pt.pointIndex;
      const imgSrc = data.images[idx];
      const idText = data.ids[idx] || "";
      const ann = data.annotations[idx] || "";
      const simVal = (data.sims && data.sims[idx] !== undefined)
        ? data.sims[idx]
        : null;

      if (!imgSrc) return;

      const safeAnn = ann.length > 500 ? ann.slice(0, 500) + '...' : ann;
      const simLine = (simVal === null)
        ? ''
        : '<div style="font-size: 11px; color: #555; margin-bottom: 2px;">' +
          'similarity: ' + simVal.toFixed(3) +
          '</div>';
      const imgHtml =
        '<div style="font-size: 12px; margin-bottom: 4px; font-weight: bold;">' + idText + '</div>' +
        simLine +
        '<div style="font-size: 11px; max-width: 280px; white-space: pre-wrap; margin-bottom: 4px;">' + safeAnn + '</div>' +
        '<img src="' + imgSrc + '" style="max-width: 240px; max-height: 240px;" />';

      tooltip.innerHTML = imgHtml;
      tooltip.style.left = (evt.event.clientX + 12) + "px";
      tooltip.style.top = (evt.event.clientY + 12) + "px";
      tooltip.style.display = "block";
    }});

    plotDiv.on("plotly_unhover", function() {{
      tooltip.style.display = "none";
    }});
  </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"t-SNE HTML (text similarity) 그래프 저장 완료: {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="학습된 인코더 체크포인트 경로 (.pt)"
    )

    # 데이터/샘플링 관련
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--json-path",
        type=str,
        default="kse/ClimateToText/stage1_preprocessed_items.json",
        help="(image, annotation, cond_name)이 들어 있는 stage1_preprocessed_items.json 경로",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="t-SNE에 사용할 최대 샘플 수",
    )
    parser.add_argument(
        "--target-cond-name",
        type=str,
        default="",
        help=(
            "특정 cond_name에 해당하는 샘플만 사용하고 싶을 때 지정 "
            "(예: 'wqa:shr6'). 지정하면 num-samples 개수만큼 "
            "해당 cond_name 샘플을 최대한 뽑습니다."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="샘플링 및 t-SNE용 랜덤 시드",
    )

    # 인코더 설정 (학습 때와 동일해야 함)
    parser.add_argument(
        "--encoder-mode",
        type=str,
        default="perceiver",
        choices=["perceiver", "perceiver_patch_mae", "resnet"],
        help="Stage1 인코더 유형",
    )
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--perceiver-num-latents", type=int, default=256)
    parser.add_argument("--perceiver-patch-size", type=int, default=16)

    # 텍스트 유사도 기반 색상 옵션
    parser.add_argument(
        "--use-text-similarity",
        action="store_true",
        help="텍스트 임베딩 기반 유사도 점수를 색깔 그라데이션으로 표시",
    )
    parser.add_argument(
        "--embedding-model-path",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help=(
            "텍스트 유사도 계산에 사용할 문장 임베딩 모델 "
            "(HF ID 또는 로컬 경로)"
        ),
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="텍스트 유사도 기준이 될 참조 샘플 인덱스 (0-based, t-SNE 샘플 기준)",
    )
    parser.add_argument(
        "--image-base-url",
        type=str,
        default="",
        help=(
            "이미지 파일을 서빙하는 HTTP 베이스 URL "
            "(예: 'http://localhost:8081'). 비우면 file:// 로 직접 접근"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    set_seed(args.seed)
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. JSON에서 cond_map 및 샘플 선택
    print(f"Loading metadata from {args.json_path} ...")
    cond_map = build_cond_map_from_json(args.json_path)
    inv_cond_map = {v: k for k, v in cond_map.items()}

    target_cond_name = args.target_cond_name.strip()
    if target_cond_name:
        print(f"cond_name='{target_cond_name}' 에 해당하는 샘플만 사용합니다.")

    items, orig_indices = select_items_for_tsne(
        args.json_path,
        cond_map,
        num_samples=args.num_samples,
        seed=args.seed,
        target_cond_name=target_cond_name,
    )
    print(
        f"t-SNE에 사용할 샘플 수: {len(items)} "
        f"(원본 JSON 인덱스 예: {orig_indices[:5]})"
    )

    # 2. 모델 로드
    print(f"Loading Model from {args.ckpt} ...")
    if args.encoder_mode == "resnet":
        from src.models import ResNetEncoder

        encoder = ResNetEncoder(
            pretrained=True,
            output_dim=args.latent_dim,
        ).to(device)
    elif args.encoder_mode == "perceiver":
        encoder = PerceiverEncoder(
            patch_size=args.perceiver_patch_size,
            latent_dim=args.latent_dim,
            num_latents=args.perceiver_num_latents,
            num_blocks=6,
            num_heads=8,
            moe_num_experts=8,
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
            moe_num_experts=8,
        ).to(device)
    else:
        raise ValueError(f"Unknown encoder_mode: {args.encoder_mode}")

    encoder.load_state_dict(torch.load(args.ckpt, map_location=device))

    # 3. 특징 및 텍스트 추출
    print("Extracting Features and texts...")
    latents, labels, texts = extract_features_from_items(
        items,
        cond_map,
        encoder,
        device,
        encoder_mode=args.encoder_mode,
        image_size=args.image_size,
    )

    # 4. t-SNE 실행
    latents_2d = run_tsne(latents, seed=args.seed)
    ckpt_dir_name = Path(args.ckpt).parent.name
    title = f"Latent Space t-SNE ({ckpt_dir_name})"

    # # 4-1. 기존 라벨 기반 색상
    # save_path_label = f"tsne_result_tab20_{ckpt_dir_name}.png"
    # plot_tsne(latents_2d, labels, inv_cond_map, save_path_label, title)

    # 4-2. 텍스트 유사도 기반 색상 (옵션)
    if args.use_text_similarity:
        if EmbeddingSimilarityScorer is None:
            print(
                "텍스트 유사도 플롯을 위해 EmbeddingSimilarityScorer를 찾지 못했습니다. "
                "script.text_similarity_embedding 모듈을 확인하세요."
            )
        else:
            # 기준 샘플 인덱스 (전체 t-SNE 샘플 기준)
            ref_global_idx = max(0, min(args.reference_index, len(texts) - 1))
            ref_item = items[ref_global_idx]
            ref_cond_name = ref_item.get("cond_name")
            ref_image_path = ref_item.get("image")
            ref_orig_json_idx = orig_indices[ref_global_idx]

            # 기준 cond_name과 같은 샘플만 선택
            same_indices = [
                i for i, it in enumerate(items) if it.get("cond_name") == ref_cond_name
            ]
            if not same_indices:
                print(
                    f"cond_name={ref_cond_name} 에 해당하는 샘플을 찾지 못했습니다. "
                    "텍스트 유사도 플롯을 건너뜁니다."
                )
            else:
                # 기준 샘플이 subset 안에서 가지는 로컬 인덱스
                try:
                    ref_local_idx = same_indices.index(ref_global_idx)
                except ValueError:
                    # 이론상 발생하지 않지만, 방어적으로 첫 번째 요소를 기준으로 사용
                    ref_local_idx = 0

                texts_same = [texts[i] for i in same_indices]
                latents_2d_same = latents_2d[same_indices, :]

                print(
                    f"Reference sample: file={ref_image_path}, "
                    f"cond_name={ref_cond_name}, "
                    f"tsne_idx={ref_global_idx}, "
                    f"orig_json_idx={ref_orig_json_idx}"
                )
                print(
                    f"텍스트 유사도 플롯에 사용할 샘플 수: {len(texts_same)} "
                    f"(동일 cond_name 기준)"
                )

                sim_scores = compute_text_similarities(
                    texts_same,
                    ref_index=ref_local_idx,
                    model_name_or_path=args.embedding_model_path,
                    device=device,
                )
                save_path_sim = f"tsne_text_similarity_{ref_cond_name}_{ckpt_dir_name}.png"
                plot_tsne_with_similarity(
                    latents_2d_same,
                    sim_scores,
                    save_path_sim,
                    title + f" (Text similarity - {ref_cond_name})",
                    ref_index=ref_local_idx,
                )

                # HTML 인터랙티브 버전 (마우스 오버 시 md1636 등 ID 표시)
                ids_same = [
                    extract_image_id_from_path(items[i].get("image", ""))
                    for i in same_indices
                ]
                annotations_same = [texts[i] for i in same_indices]
                image_paths_same = [items[i].get("image", "") for i in same_indices]
                html_path = (
                    f"tsne_text_similarity_{ref_cond_name}_{ckpt_dir_name}.html"
                )
                save_tsne_html_with_similarity(
                    latents_2d_same,
                    sim_scores,
                    ids_same,
                    annotations_same,
                    image_paths_same,
                    ref_local_idx,
                    html_path,
                    title + f" (Text similarity - {ref_cond_name})",
                    base_url=args.image_base_url,
                )


if __name__ == "__main__":
    main()
