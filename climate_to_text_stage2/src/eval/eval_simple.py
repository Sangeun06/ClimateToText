# twkim/climate_to_text_stage2/eval_simple.py

"""
간단한 Stage2 모델 평가 스크립트
--------------------------------

다음 네 가지 모델을 동일한 WeatherQA SUMMARY-only(three types) 데이터셋으로 평가한다.

  1) pretrained LLM + ResNet Stage2 prefix (LLM 미세튜닝 없음)
  2) ResNet image encoder로 학습된 Stage2 (LoRA)
  3) Stage1 encoder + classifier (frozen) 기반 Stage2 (LoRA)
  4) Stage1 encoder 마지막 블록 몇 개를 finetune한 Stage2 (LoRA)

각 모델에 대해:
  - 임베딩 기반 텍스트 유사도 평균 (GT vs generated)
  - LLM-as-a-judge 기반 유사도 평균
  - 일부 샘플의 GT / 각 모델 출력 텍스트 예시

가능한 한 하이퍼파라미터를 코드 내부에서 기본값으로 지정해서,
거의 인자 없이 실행할 수 있도록 구성했다.

예시:

    CUDA_VISIBLE_DEVICES=0 python -m twkim.climate_to_text_stage2.eval_simple
"""

import argparse
import json
import random
import os
from typing import Dict, List, Tuple

import torch
from torchvision import transforms

from src.data.dataset import default_image_loader
from src.inference.inference_stage2 import load_stage2_model
from src.inference.inference_stage2_stage1 import load_stage2_model_stage1
from src.eval.text_similarity_embedding import EmbeddingSimilarityScorer
from src.eval.text_similarity_llm_judge import LLMSimilarityJudge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained vs 3 Stage2 variants (ResNet / Stage1 frozen / Stage1 finetuned)."
    )

    # 데이터/모델 경로: 대부분 현재 실험 환경에 맞는 기본값을 제공
    parser.add_argument(
        "--json-path",
        type=str,
        default="kse/ClimateToText/stage1_weatherqa_summary_only_three_types.json",
        help="(image, annotation, cond_name) 샘플이 들어 있는 JSON 경로",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/home/agi592/models/mistral-7b",
        help="pretrained Mistral-7B가 저장된 로컬 경로",
    )
    parser.add_argument(
        "--embedding-model-path",
        type=str,
        default="/home/agi592/models/all-mpnet-base-v2",
        help="문장 임베딩 기반 유사도 metric에 사용할 텍스트 인코더 경로",
    )

    # 세 가지 Stage2 체크포인트 경로 (질문에서 준 디렉토리를 기본값으로 사용)
    parser.add_argument(
        "--resnet-checkpoint-dir",
        type=str,
        default="twkim/checkpoints/climate_to_text_stage2_wqa_summary_only_ep1_r32_3types",
        help="ResNet Stage2 LoRA 체크포인트 디렉토리",
    )
    parser.add_argument(
        "--stage1-checkpoint-dir",
        type=str,
        default="twkim/checkpoints/climate_to_text_stage2_wqa_summary_only_ep1_r32_stage1",
        help="Stage1 encoder(frozen) 기반 Stage2 LoRA 체크포인트 디렉토리",
    )
    parser.add_argument(
        "--stage1-ft-checkpoint-dir",
        type=str,
        default="twkim/checkpoints/climate_to_text_stage2_wqa_summary_only_ep1_r32_stage1_ft2",
        help="Stage1 encoder 부분적으로 finetune한 Stage2 LoRA 체크포인트 디렉토리",
    )

    # Stage1 encoder / classifier 가중치 경로 (Stage1 기반 모델에만 필요)
    parser.add_argument(
        "--stage1-encoder-ckpt",
        type=str,
        default="/home/agi592/kse/ClimateToText/stage1_mtl_weatherqa_three_patch/stage1_vision_encoder_mae.pt",
        help="Stage1 이미지 인코더 가중치 경로",
    )
    parser.add_argument(
        "--stage1-classifier-ckpt",
        type=str,
        default="/home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt",
        help="Stage1 이미지 분류기 가중치 경로",
    )

    # 평가 관련 옵션 (기본값으로 충분히 쓸 수 있게 최소화)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="최대 평가 샘플 수 (JSON 순서 기준 상위 N개, 이미지 파일 없으면 건너뜀)",
    )
    parser.add_argument(
        "--num-print-samples",
        type=int,
        default=3,
        help="콘솔에 예시로 출력할 샘플 수",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="이미지 인코더 입력 해상도",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="텍스트 생성 시 최대 토큰 수",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="beam search 폭 (1이면 greedy)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=4,
        help="반복 억제를 위한 no_repeat_ngram_size (0이면 비활성화)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="토큰 반복 패턴을 줄이기 위한 repetition_penalty (1.0이면 비활성화)",
    )

    # 어떤 metric을 사용할지 선택 (GPU 메모리 부담을 줄이고 싶을 때 single metric 선택)
    parser.add_argument(
        "--metric",
        type=str,
        choices=["both", "embedding", "judge"],
        default="both",
        help="평가에 사용할 metric 종류 선택 (both / embedding / judge)",
    )

    parser.add_argument(
        "--tokenizer-max-length",
        type=int,
        default=2048,
        help="프롬프트 토크나이저 max_length 강제 설정 (truncation warning 제거용)",
    )

    parser.add_argument(
        "--plot-path",
        type=str,
        default="",
        help="모델별 점수 분포를 저장할 PNG 경로 (비우면 플롯 생성 안 함)",
    )
    parser.add_argument(
        "--sample-plot-path",
        type=str,
        default="",
        help="샘플별 점수(샘플 인덱스 vs score)를 저장할 PNG 경로",
    )

    return parser.parse_args()


def build_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_items(json_path: str, max_samples: int) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    selected: List[Dict] = []
    for it in items:
        img_path = it.get("image")
        if not img_path or not os.path.exists(img_path):
            continue
        if "annotation" not in it or "cond_name" not in it:
            continue
        selected.append(it)
        if len(selected) >= max_samples:
            break
    return selected


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    items = load_items(args.json_path, args.max_samples)
    print(f"Loaded {len(items)} items from {args.json_path} (up to {args.max_samples})")
    if not items:
        print("No valid items found; check json-path and image paths.")
        return

    image_transform = build_image_transform(args.image_size)

    # --- 어떤 metric을 쓸지 결정 ---
    use_emb = args.metric in ("both", "embedding")
    use_judge = args.metric in ("both", "judge")

    # 모델별 생성 결과 저장용: model_name -> List[str|None] (items 인덱스별)
    outputs_by_model: Dict[str, List[str]] = {}
    valid_per_model: Dict[str, int] = {}

    # 프롬프트는 모든 모델 동일하게 사용
    generation_prompt = (
        "Describe the weather situation shown in this meteorological image. "
        "Write a short, clear summary in complete sentences."
    )

    # ------------------------------------------------------------------
    # 모델 로더 정의 (존재하는 체크포인트만 대상)
    # ------------------------------------------------------------------
    model_loaders = []
    # Stage1 계열에서 cond_name_to_id.json이 없을 경우 대비: dataset에서 만들 수 있는 매핑
    fallback_cond_map = None
    if items:
        uniq_cond = sorted({it.get("cond_name") for it in items if it.get("cond_name")})
        fallback_cond_map = {c: i for i, c in enumerate(uniq_cond)}

    def ensure_cond_map(checkpoint_dir: str) -> None:
        """
        Stage1 계열 ckpt에 cond_name_to_id.json이 없으면,
        데이터셋 기반 fallback 매핑을 만들어 넣는다.
        (ResNet 경로에는 손대지 않음)
        """
        if not fallback_cond_map:
            return
        mapping_path = os.path.join(checkpoint_dir, "cond_name_to_id.json")
        if not os.path.exists(mapping_path):
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                with open(mapping_path, "w", encoding="utf-8") as f:
                    json.dump(fallback_cond_map, f, ensure_ascii=False, indent=2)
                print(f"  (created fallback cond_name_to_id.json in {checkpoint_dir})")
            except Exception as e:
                print(f"  (warn) failed to create fallback cond map in {checkpoint_dir}: {e}")

    if os.path.isdir(args.resnet_checkpoint_dir):
        # ResNet prefix + pretrained LLM
        model_loaders.append(
            (
                "pretrained_resnet",
                lambda: load_stage2_model(
                    checkpoint_dir=args.resnet_checkpoint_dir,
                    base_model_path=args.base_model_path,
                    use_lora=False,
                    use_pretrained_llm_only=True,
                    device=device,
                ),
            )
        )
        # ResNet + LoRA
        model_loaders.append(
            (
                "resnet_lora",
                lambda: load_stage2_model(
                    checkpoint_dir=args.resnet_checkpoint_dir,
                    base_model_path=args.base_model_path,
                    use_lora=True,
                    use_pretrained_llm_only=False,
                    device=device,
                ),
            )
        )
    else:
        print(f"(skip) resnet checkpoint not found: {args.resnet_checkpoint_dir}")

    if os.path.isdir(args.stage1_checkpoint_dir):
        ensure_cond_map(args.stage1_checkpoint_dir)
        model_loaders.append(
            (
                "stage1_frozen",
                lambda: load_stage2_model_stage1(
                    checkpoint_dir=args.stage1_checkpoint_dir,
                    base_model_path=args.base_model_path,
                    use_lora=True,
                    use_pretrained_llm_only=False,
                    device=device,
                    stage1_encoder_ckpt=args.stage1_encoder_ckpt,
                    stage1_classifier_ckpt=args.stage1_classifier_ckpt,
                ),
            )
        )
    else:
        print(f"(skip) stage1 checkpoint not found: {args.stage1_checkpoint_dir}")

    if os.path.isdir(args.stage1_ft_checkpoint_dir):
        ensure_cond_map(args.stage1_ft_checkpoint_dir)
        model_loaders.append(
            (
                "stage1_ft",
                lambda: load_stage2_model_stage1(
                    checkpoint_dir=args.stage1_ft_checkpoint_dir,
                    base_model_path=args.base_model_path,
                    use_lora=True,
                    use_pretrained_llm_only=False,
                    device=device,
                    stage1_encoder_ckpt=args.stage1_encoder_ckpt,
                    stage1_classifier_ckpt=args.stage1_classifier_ckpt,
                ),
            )
        )
    else:
        print(f"(skip) stage1-ft checkpoint not found: {args.stage1_ft_checkpoint_dir}")

    if not model_loaders:
        print("No valid checkpoints found. Nothing to evaluate.")
        return

    # ------------------------------------------------------------------
    # 모델별로 순차 로딩 → 전체 샘플 생성 → 문자열만 보관 → 모델 언로드
    # ------------------------------------------------------------------
    for model_name, loader_fn in model_loaders:
        print(f"\n[Load] {model_name}")
        try:
            model, tokenizer_loaded, cond_name_to_id = loader_fn()
        except FileNotFoundError as e:
            # cond_name_to_id.json 등 필수 파일이 없는 경우 건너뜀
            print(f"  (skip) {model_name}: {e}")
            continue
        # truncation warning 방지를 위해 max_length 설정
        if hasattr(tokenizer_loaded, "model_max_length"):
            tokenizer_loaded.model_max_length = args.tokenizer_max_length
        outputs = [None] * len(items)
        valid = 0

        for idx, it in enumerate(items):
            image_path = it["image"]
            cond_name = it["cond_name"]
            if cond_name not in cond_name_to_id:
                continue

            img = default_image_loader(image_path)
            pixel_values = image_transform(img).unsqueeze(0).to(device)
            cond_id = cond_name_to_id[cond_name]
            cond_ids = torch.tensor([cond_id], dtype=torch.long, device=device)

            with torch.no_grad():
                gen_text = model.generate(
                    pixel_values=pixel_values,
                    cond_ids=cond_ids,
                    tokenizer=tokenizer_loaded,
                    generation_prompt=generation_prompt,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                )[0]
            outputs[idx] = gen_text
            valid += 1

        outputs_by_model[model_name] = outputs
        valid_per_model[model_name] = valid

        # 메모리 해제
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 메트릭 계산 (이 시점에는 Stage2 LLM이 내려가 있으므로 judge를 GPU에 올려도 됨)
    # ------------------------------------------------------------------
    emb_scorer = None
    llm_judge = None
    if use_emb:
        # GPU 메모리 부족을 방지하기 위해 기본은 CPU로 로드
        emb_scorer = EmbeddingSimilarityScorer(
            model_name_or_path=args.embedding_model_path,
            device=str(device),
        )
    if use_judge:
        llm_judge = LLMSimilarityJudge(
            model_name_or_path=args.base_model_path,
            device=str(device),
        )

    sums_emb: Dict[str, float] = {name: 0.0 for name, _ in model_loaders}
    sums_llm: Dict[str, float] = {name: 0.0 for name, _ in model_loaders}
    counts_emb: Dict[str, int] = {name: 0 for name, _ in model_loaders}
    counts_llm: Dict[str, int] = {name: 0 for name, _ in model_loaders}
    all_emb_scores: Dict[str, List[Tuple[int, float]]] = {name: [] for name, _ in model_loaders}
    all_llm_scores: Dict[str, List[Tuple[int, float]]] = {name: [] for name, _ in model_loaders}

    for idx, it in enumerate(items):
        gt = it.get("annotation", "").strip()
        if not gt:
            continue
        for model_name in outputs_by_model.keys():
            out = outputs_by_model[model_name][idx]
            if out is None:
                continue
            if emb_scorer is not None:
                score = emb_scorer.score(gt, out)
                sums_emb[model_name] += score
                counts_emb[model_name] += 1
                all_emb_scores[model_name].append((idx, score))
            if llm_judge is not None:
                score = llm_judge.score(gt, out)
                sums_llm[model_name] += score
                counts_llm[model_name] += 1
                all_llm_scores[model_name].append((idx, score))

    total_samples = len(items)
    print(f"\nEvaluated {total_samples} samples (per-model valid counts may differ).")

    if emb_scorer is not None:
        print("\n=== Average Embedding Similarity (GT vs output) ===")
        for name, total in sums_emb.items():
            cnt = max(counts_emb[name], 1)
            print(f"{name:20s}: {total / cnt:.4f}  (#samples={counts_emb[name]})")

    if llm_judge is not None:
        print("\n=== Average LLM-Judge Similarity (GT vs output) ===")
        for name, total in sums_llm.items():
            cnt = max(counts_llm[name], 1)
            print(f"{name:20s}: {total / cnt:.4f}  (#samples={counts_llm[name]})")

    # ------------------------------------------------------------------
    # 예시 출력 (valid 인덱스 중 랜덤 샘플링)
    # ------------------------------------------------------------------
    valid_indices = [
        idx
        for idx, it in enumerate(items)
        if any(outs[idx] is not None for outs in outputs_by_model.values())
    ]
    if valid_indices and args.num_print_samples > 0:
        sample_indices = random.sample(
            valid_indices, k=min(args.num_print_samples, len(valid_indices))
        )
    else:
        sample_indices = []

    examples: List[Tuple[str, Dict[str, str]]] = []
    for idx in sample_indices:
        it = items[idx]
        row_outputs: Dict[str, str] = {}
        for name, outs in outputs_by_model.items():
            if outs[idx] is not None:
                row_outputs[name] = outs[idx]
        examples.append(
            (
                f"{it['image']} | cond_name={it['cond_name']}",
                {"gt": it.get("annotation", "").strip(), **row_outputs},
            )
        )

    if examples:
        print("\n=== Example Generations ===")
        for idx, (meta, texts) in enumerate(examples, start=1):
            print(f"\n[Sample {idx}] {meta}")
            print("--- Ground Truth ---")
            print(texts.get("gt", ""))
            for name in outputs_by_model.keys():
                if name in texts:
                    print(f"--- {name} ---")
                    print(texts[name])

    # ------------------------------------------------------------------
    # 점수 분포 플롯 (옵션)
    # ------------------------------------------------------------------
    if args.plot_path:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot.")
        else:
            plt.figure(figsize=(10, 6))
            plot_count = 0

            if emb_scorer is not None:
                plot_count += 1
                plt.subplot(1, 2 if llm_judge is not None else 1, plot_count)
                data = [
                    [s for _, s in all_emb_scores[name]]
                    for name, _ in model_loaders
                    if all_emb_scores[name]
                ]
                labels = [name for name, _ in model_loaders if all_emb_scores[name]]
                if data:
                    # 중앙값 선 없이 평균만 표시하기 위해 showmeans만 사용하고 medianprops를 숨김
                    plt.boxplot(
                        data,
                        tick_labels=labels,  # Matplotlib>=3.9
                        showfliers=False,
                        showmeans=True,
                        meanline=True,
                        meanprops=dict(color="red", linewidth=1.5),
                        medianprops=dict(color="none"),
                    )
                    plt.title("Embedding similarity")
                    plt.ylabel("score")
                    plt.xticks(rotation=15, ha="right")

            if llm_judge is not None:
                plot_count += 1
                plt.subplot(1, 2 if emb_scorer is not None else 1, plot_count)
                data = [
                    [s for _, s in all_llm_scores[name]]
                    for name, _ in model_loaders
                    if all_llm_scores[name]
                ]
                labels = [name for name, _ in model_loaders if all_llm_scores[name]]
                if data:
                    plt.boxplot(
                        data,
                        tick_labels=labels,  # Matplotlib>=3.9
                        showfliers=False,
                        showmeans=True,
                        meanline=True,
                        meanprops=dict(color="red", linewidth=1.5),
                        medianprops=dict(color="none"),
                    )
                    plt.title("LLM judge similarity")
                    plt.ylabel("score")
                    plt.xticks(rotation=15, ha="right")

            if plot_count > 0:
                plot_dir = os.path.dirname(args.plot_path)
                if plot_dir:
                    os.makedirs(plot_dir, exist_ok=True)
                plt.tight_layout()
                plt.savefig(args.plot_path, dpi=150)
                print(f"\nSaved score plot to: {args.plot_path}")
            else:
                print("No scores available for plotting.")

    # ------------------------------------------------------------------
    # 샘플별 점수 플롯 (샘플 인덱스 vs score, 모델별 색상)
    # ------------------------------------------------------------------
    if args.sample_plot_path:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping sample plot.")
        else:
            # 선택된 metric 중 하나만 그린다 (embedding 우선, 없으면 judge)
            metric_data = None
            metric_title = ""
            if emb_scorer is not None:
                metric_data = all_emb_scores
                metric_title = "Embedding similarity"
            elif llm_judge is not None:
                metric_data = all_llm_scores
                metric_title = "LLM judge similarity"

            if metric_data:
                plt.figure(figsize=(10, 5))
                for name, scores in metric_data.items():
                    if not scores:
                        continue
                    xs = [idx for idx, _ in scores]
                    ys = [s for _, s in scores]
                    order = sorted(range(len(xs)), key=lambda i: xs[i])
                    plt.plot([xs[i] for i in order], [ys[i] for i in order], alpha=0.8, label=name)

                plt.title(f"{metric_title} per sample")
                plt.xlabel("sample index")
                plt.ylabel("score")
                plt.legend()
                plt.tight_layout()
                plot_dir = os.path.dirname(args.sample_plot_path)
                if plot_dir:
                    os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(args.sample_plot_path, dpi=150)
                print(f"\nSaved sample-wise plot to: {args.sample_plot_path}")
            else:
                print("No metric selected for sample-wise plot.")


if __name__ == "__main__":
    main()
