"""
Run all 4 models on the same sampled WeatherQA items and save outputs in sample.json format.

Models:
- pretrained (text-only)
- baseline Stage2
- MTL Stage2
- Frontier Stage2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
    
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from climate_to_text_stage2.dataset import default_image_loader
from climate_to_text_stage2.mtl_weatherqa.data_utils import build_image_transform, resolve_image_path
from climate_to_text_stage2.mtl_weatherqa.model_utils import load_stage2_lora, load_pretrained_llm
from climate_to_text_stage2.text_similarity_embedding import EmbeddingSimilarityScorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pretrained + 3 Stage2 models on the same samples and save JSON.")
    p.add_argument("--json-path", type=str, required=True, help="Gemini JSON (global_summary, elements)")
    p.add_argument("--image-root", type=str, required=True, help="Root dir to resolve elements.image_rel_path")
    p.add_argument("--num-times", type=int, default=1, help="How many time slots (cases) to sample. Approx total = 20 x num_times.")
    p.add_argument("--seed", type=int, default=42, help="Optional seed for reproducible sampling.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-new-tokens", type=int, default=80, help="Stage2 max new tokens (baseline/mtl/frontier).")
    p.add_argument("--pretrained-max-new-tokens", type=int, default=128, help="Text-only max new tokens.")
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument(
        "--prompt",
        type=str,
        default="Forecast summary in 2-3 sentences. Focus on hazards, region, and timing.",
    )
    p.add_argument("--save-path", type=str, default="evaluation/sample.json", help="Where to write results JSON.")
    p.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda:0).")

    # Model paths
    p.add_argument("--base-model-path", type=str, required=True, help="Pretrained LLM path (LoRA base).")
    p.add_argument(
        "--baseline-ckpt",
        type=str,
        required=True,
        help="Baseline Stage2 checkpoint dir (LoRA).",
    )
    p.add_argument(
        "--mtl-ckpt",
        type=str,
        required=True,
        help="MTL Stage2 checkpoint dir (LoRA).",
    )
    p.add_argument(
        "--frontier-ckpt",
        type=str,
        required=True,
        help="Frontier Stage2 checkpoint dir (LoRA).",
    )
    p.add_argument(
        "--stage1-encoder-ckpt",
        type=str,
        required=True,
        help="Stage1 encoder ckpt used for baseline/mtl.",
    )
    p.add_argument(
        "--stage1-classifier-ckpt",
        type=str,
        required=True,
        help="Stage1 classifier ckpt.",
    )
    p.add_argument(
        "--frontier-stage1-encoder-ckpt",
        type=str,
        required=True,
        help="Stage1 encoder ckpt for frontier.",
    )
    p.add_argument(
        "--baseline-encoder-mode",
        type=str,
        default="perceiver_patch_mae",
        choices=["clip", "perceiver_patch_mae", "perceiver", "resnet", "vit"],
    )
    p.add_argument(
        "--mtl-encoder-mode",
        type=str,
        default="perceiver_patch_mae",
        choices=["clip", "perceiver_patch_mae", "perceiver", "resnet", "vit"],
    )
    p.add_argument(
        "--frontier-encoder-mode",
        type=str,
        default="clip",
        choices=["clip", "perceiver_patch_mae", "perceiver", "resnet", "vit"],
    )

    p.add_argument(
        "--embedding-model-path",
        type=str,
        default=None,
        help="Optional text encoder to compute emb_scores (GT vs generations).",
    )
    return p.parse_args()


def generate_stage2(
    model,
    tokenizer,
    pixel_values: torch.Tensor,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
) -> str:
    cond_ids = torch.zeros(pixel_values.size(0), dtype=torch.long, device=pixel_values.device)
    with torch.no_grad():
        out = model.generate(
            pixel_values=pixel_values,
            cond_ids=cond_ids,
            tokenizer=tokenizer,
            generation_prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )[0]
    return out


def generate_pretrained(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text.strip()


def sample_time_slots(json_path: str, image_root: str, num_times: int, seed: int | None = None) -> List[Dict]:
    """Randomly pick `num_times` cases (time slots) and return all their elements (≈20 per case)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    chosen = indices[:num_times]

    samples: List[Dict] = []
    for idx in chosen:
        case = data[idx]
        summary = (case.get("global_summary") or "").strip()
        for el in case.get("elements", []):
            cond = el.get("cond_name") or el.get("type_name")
            rel = el.get("image_rel_path")
            cap = (el.get("caption") or "").strip()
            if not (summary and cond and rel):
                continue
            img_path = resolve_image_path(rel, image_root)
            if not img_path:
                continue
            samples.append(
                {
                    "image": img_path,
                    "cond_name": cond,
                    "summary": summary,
                    "caption": cap,
                    "case_index": idx,
                }
            )
    return samples


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    samples = sample_time_slots(args.json_path, args.image_root, args.num_times, seed=args.seed)
    print(f"Selected {len(samples)} samples from {args.json_path} (time slots={args.num_times})")
    if not samples:
        return

    transform = build_image_transform(args.image_size, imagenet_norm=False)

    # Preload images once (CPU tensors)
    pixel_cache: List[torch.Tensor] = []
    for s in samples:
        img = default_image_loader(s["image"])
        pv = transform(img).unsqueeze(0)  # shape [1, 3, H, W] on CPU
        pixel_cache.append(pv)

    scorer = None
    if args.embedding_model_path:
        scorer = EmbeddingSimilarityScorer(model_name_or_path=args.embedding_model_path, device="cpu")
        print(f"Embedding scorer loaded: {args.embedding_model_path}")

    results: List[Dict] = []
    emb_scores = {"pretrained": [], "baseline": [], "mtl": [], "frontier": []}

    # ----- Pretrained text-only -----
    print("[Load] pretrained text-only LLM")
    pt_model, pt_tok = load_pretrained_llm(args.base_model_path, device=device)
    pretrained_outs = []
    for s in samples:
        prompt = f"Condition: {s['cond_name']}.\n{args.prompt}"
        pretrained_outs.append(
            generate_pretrained(
                pt_model,
                pt_tok,
                prompt=prompt,
                max_new_tokens=args.pretrained_max_new_tokens,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )
        )
    del pt_model, pt_tok
    torch.cuda.empty_cache()

    # ----- Baseline Stage2 -----
    print("[Load] baseline Stage2")
    baseline_model, baseline_tok = load_stage2_lora(
        checkpoint_dir=args.baseline_ckpt,
        base_model_path=args.base_model_path,
        encoder_ckpt=args.stage1_encoder_ckpt,
        classifier_ckpt=args.stage1_classifier_ckpt,
        device=device,
        num_image_tokens=4,
        encoder_mode=args.baseline_encoder_mode,
    )
    baseline_outs = []
    for pv in pixel_cache:
        baseline_outs.append(
            generate_stage2(
                baseline_model,
                baseline_tok,
                pixel_values=pv.to(device),
                prompt=f"Condition: {samples[len(baseline_outs)]['cond_name']}\n{args.prompt}",
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )
        )
    del baseline_model, baseline_tok
    torch.cuda.empty_cache()

    # ----- MTL Stage2 -----
    print("[Load] MTL Stage2")
    mtl_model, mtl_tok = load_stage2_lora(
        checkpoint_dir=args.mtl_ckpt,
        base_model_path=args.base_model_path,
        encoder_ckpt=args.stage1_encoder_ckpt,
        classifier_ckpt=args.stage1_classifier_ckpt,
        device=device,
        num_image_tokens=4,
        encoder_mode=args.mtl_encoder_mode,
    )
    mtl_outs = []
    for pv in pixel_cache:
        mtl_outs.append(
            generate_stage2(
                mtl_model,
                mtl_tok,
                pixel_values=pv.to(device),
                prompt=f"Condition: {samples[len(mtl_outs)]['cond_name']}\n{args.prompt}",
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )
        )
    del mtl_model, mtl_tok
    torch.cuda.empty_cache()

    # ----- Frontier Stage2 -----
    print("[Load] Frontier Stage2")
    frontier_model, frontier_tok = load_stage2_lora(
        checkpoint_dir=args.frontier_ckpt,
        base_model_path=args.base_model_path,
        encoder_ckpt=args.frontier_stage1_encoder_ckpt,
        classifier_ckpt=args.stage1_classifier_ckpt,
        device=device,
        num_image_tokens=4,
        encoder_mode=args.frontier_encoder_mode,
    )
    frontier_outs = []
    for pv in pixel_cache:
        frontier_outs.append(
            generate_stage2(
                frontier_model,
                frontier_tok,
                pixel_values=pv.to(device),
                prompt=f"Condition: {samples[len(frontier_outs)]['cond_name']}\n{args.prompt}",
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )
        )
    del frontier_model, frontier_tok
    torch.cuda.empty_cache()

    # ----- Collect results -----
    for s, gen_pt, gen_base, gen_mtl, gen_front in zip(samples, pretrained_outs, baseline_outs, mtl_outs, frontier_outs):
        gt = s.get("summary", "")
        cap_gt = s.get("caption", "")
        res = {
            "image": s["image"],
            "cond_name": s.get("cond_name", ""),
            "gt": gt,
            "caption_gt": cap_gt,
            "pretrained": gen_pt,
            "baseline": gen_base,
            "mtl": gen_mtl,
            "frontier": gen_front,
        }
        if scorer:
            emb_scores["pretrained"].append(scorer.score(gt, gen_pt))
            emb_scores["baseline"].append(scorer.score(gt, gen_base))
            emb_scores["mtl"].append(scorer.score(gt, gen_mtl))
            emb_scores["frontier"].append(scorer.score(gt, gen_front))
        results.append(res)

    payload: Dict = {"emb_scores": emb_scores if scorer else {}, "results": results}
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved outputs to {save_path}")


if __name__ == "__main__":
    main()
