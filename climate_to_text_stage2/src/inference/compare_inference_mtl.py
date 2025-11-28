"""
MTL vs baseline vs pretrained 비교 스크립트

- pretrained LLM (텍스트-only)
- baseline Stage2 (예: 기존 image→text finetune ckpt, Stage1 기반)
- MTL Stage2 (이번 학습된 ckpt)

여러 샘플에 대해 생성 결과와 embedding similarity를 비교한다.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.data.dataset import default_image_loader
from src.models.stage1_stage2_integration import Stage1ImageEncoderForStage2, ImageToTextModelStage1
from src.eval.text_similarity_embedding import EmbeddingSimilarityScorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare pretrained vs baseline vs MTL Stage2 outputs.")
    p.add_argument("--json-path", type=str, required=True, help="(image, annotation, cond_name) JSON 경로")
    p.add_argument(
        "--image-root",
        type=str,
        default="",
        help="gemini json처럼 image_rel_path를 쓸 때 붙일 루트 (절대/상대 경로). 기존 image 필드를 그대로 쓴다면 비워둠.",
    )
    p.add_argument("--num-samples", type=int, default=3, help="서로 다른 cond_name 샘플 몇 개를 비교할지")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--repetition-penalty", type=float, default=1.1)

    # 모델 경로들
    p.add_argument("--pretrained-model-path", type=str, required=True, help="pretrained LLM 경로 (텍스트-only)")
    p.add_argument("--baseline-ckpt", type=str, required=True, help="기존 Stage2 finetune ckpt (Stage1 기반)")
    p.add_argument("--mtl-ckpt", type=str, required=True, help="MTL Stage2 ckpt")
    p.add_argument("--stage1-encoder-ckpt", type=str, required=True, help="Stage1 encoder ckpt")
    p.add_argument("--stage1-classifier-ckpt", type=str, required=True, help="Stage1 classifier ckpt")

    # 프롬프트
    p.add_argument(
        "--prompt",
        type=str,
        default="Describe the weather situation shown in this meteorological image.",
    )

    # 임베딩 모델
    p.add_argument(
        "--embedding-model-path",
        type=str,
        default="/home/agi592/models/all-mpnet-base-v2",
        help="embedding similarity에 사용할 모델",
    )
    p.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="JSON으로 결과를 저장할 경로 (옵션). 지정하면 {image, cond_name, gt, caption_gt, pretrained, baseline, mtl, emb_scores}를 기록.",
    )

    return p.parse_args()


def select_samples(json_path: str, image_root: str, num_samples: int) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # image_root 아래에 WeatherQA_MD_* 하위 디렉토리가 있는 경우를 모두 후보로 사용
    candidate_roots = [image_root] if image_root else [""]
    for sub in ["WeatherQA_MD_2014-2019", "WeatherQA_MD_2020"]:
        full = os.path.join(image_root, sub) if image_root else sub
        if os.path.isdir(full):
            candidate_roots.append(full)

    def resolve_path(rel_path: str) -> str:
        rel_norm = rel_path.lstrip("./")
        for base in candidate_roots:
            cand = os.path.join(base, rel_norm) if base else rel_norm
            if os.path.exists(cand):
                return cand
        return rel_path  # fallback

    selected = []
    seen = set()

    # Case 1) 기존 형식: [{image, cond_name, annotation}, ...]
    if items and isinstance(items[0], dict) and "image" in items[0]:
        for it in items:
            cond = it.get("cond_name")
            img = it.get("image")
            if not cond or not img:
                continue
            if cond in seen:
                continue
            if not os.path.exists(img):
                continue
            seen.add(cond)
            selected.append(it)
            if len(selected) >= num_samples:
                break
        return selected

    # Case 2) gemini 형식: [{global_summary, elements:[{image_rel_path, cond_name, ...}, ...]}, ...]
    for case in items:
        summary = case.get("global_summary", "").strip()
        elements = case.get("elements", [])
        for elem in elements:
            cond = elem.get("cond_name") or elem.get("type_name")
            rel = elem.get("image_rel_path")
            cap = elem.get("caption", "").strip()
            if not cond or not rel:
                continue
            if cond in seen:
                continue
            img = resolve_path(rel)
            if not os.path.exists(img):
                continue
            seen.add(cond)
            selected.append(
                {
                    "image": img,
                    "cond_name": cond,
                    "annotation": summary,
                    "caption": cap,
                }
            )
            if len(selected) >= num_samples:
                return selected

    return selected


def load_stage1_model(checkpoint_dir: str, base_model_path: str, encoder_ckpt: str, classifier_ckpt: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    llm = PeftModel.from_pretrained(base_llm, checkpoint_dir)
    llm.to(device)

    image_encoder = Stage1ImageEncoderForStage2(
        encoder_ckpt_path=encoder_ckpt,
        classifier_ckpt_path=classifier_ckpt,
        device=device,
        freeze=True,
    )

    finetuned_enc_path = os.path.join(checkpoint_dir, "stage1_encoder_finetuned.pt")
    if os.path.exists(finetuned_enc_path) and hasattr(image_encoder, "pipeline") and hasattr(image_encoder.pipeline, "encoder"):
        try:
            enc_state = torch.load(finetuned_enc_path, map_location=device, weights_only=True)
        except TypeError:
            enc_state = torch.load(finetuned_enc_path, map_location=device)
        try:
            image_encoder.pipeline.encoder.load_state_dict(enc_state, strict=False)
            print(f"Loaded finetuned Stage1 encoder from {finetuned_enc_path}")
        except Exception as e:
            print(f"Warning: failed to load finetuned Stage1 encoder: {e}")

    model = ImageToTextModelStage1(
        llm=llm,
        image_encoder=image_encoder,
        num_image_tokens=4,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(
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


def generate_pretrained_text(
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


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = select_samples(args.json_path, args.image_root, args.num_samples)
    print(f"Selected {len(samples)} samples from {args.json_path}")
    if not samples:
        return

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    emb_scorer = EmbeddingSimilarityScorer(
        model_name_or_path=args.embedding_model_path,
        device="cpu",
    )

    # 이미지 로드 한 번만 해두기 (CPU)
    pixel_cache: List[torch.Tensor] = []
    for it in samples:
        img = default_image_loader(it["image"])
        pv = transform(img).unsqueeze(0)  # CPU tensor
        pixel_cache.append(pv)

    results: List[Dict[str, str]] = []
    emb_scores = {"pretrained": [], "baseline": [], "mtl": []}

    # ----- Baseline Stage2 -----
    print("\n[Load] baseline Stage2 (LoRA + Stage1 encoder)")
    baseline_model, baseline_tok = load_stage1_model(
        checkpoint_dir=args.baseline_ckpt,
        base_model_path=args.pretrained_model_path,
        encoder_ckpt=args.stage1_encoder_ckpt,
        classifier_ckpt=args.stage1_classifier_ckpt,
        device=device,
    )
    baseline_outs = []
    for pv in pixel_cache:
        gen = generate_text(
            baseline_model,
            baseline_tok,
            pixel_values=pv.to(device),
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
        )
        baseline_outs.append(gen)
    del baseline_model, baseline_tok
    torch.cuda.empty_cache()

    # ----- MTL Stage2 -----
    print("[Load] MTL Stage2 (LoRA + Stage1 encoder)")
    mtl_model, mtl_tok = load_stage1_model(
        checkpoint_dir=args.mtl_ckpt,
        base_model_path=args.pretrained_model_path,
        encoder_ckpt=args.stage1_encoder_ckpt,
        classifier_ckpt=args.stage1_classifier_ckpt,
        device=device,
    )
    mtl_outs = []
    for pv in pixel_cache:
        gen = generate_text(
            mtl_model,
            mtl_tok,
            pixel_values=pv.to(device),
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
        )
        mtl_outs.append(gen)
    del mtl_model, mtl_tok
    torch.cuda.empty_cache()

    # ----- Pretrained text-only -----
    print("[Load] pretrained text-only LLM")
    pt_tok = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    if pt_tok.pad_token is None:
        pt_tok.pad_token = pt_tok.eos_token
    pt_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    pt_model.eval()

    pt_outs = []
    for _ in pixel_cache:
        pt_outs.append(
            generate_pretrained_text(
                pt_model,
                pt_tok,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )
        )
    del pt_model, pt_tok
    torch.cuda.empty_cache()

    # ----- Collect results & scores -----
    for it, gen_base, gen_mtl, gen_pt in zip(samples, baseline_outs, mtl_outs, pt_outs):
        gt = it.get("annotation", "").strip()
        cap_gt = it.get("caption", "").strip()

        emb_scores["pretrained"].append(emb_scorer.score(gt, gen_pt))
        emb_scores["baseline"].append(emb_scorer.score(gt, gen_base))
        emb_scores["mtl"].append(emb_scorer.score(gt, gen_mtl))

        results.append(
            {
                "image": it["image"],
                "cond_name": it.get("cond_name", ""),
                "gt": gt,
                "caption_gt": cap_gt,
                "pretrained": gen_pt,
                "baseline": gen_base,
                "mtl": gen_mtl,
            }
        )

    print("\n=== Embedding Similarity (GT vs output) ===")
    for k, v in emb_scores.items():
        if v:
            print(f"{k:10s}: mean={sum(v)/len(v):.4f} (n={len(v)})")

    print("\n=== Sample Outputs ===")
    for i, res in enumerate(results, 1):
        print(f"\n[Sample {i}] {res['image']} | cond_name={res['cond_name']}")
        print("--- Global Summary (GT) ---")
        print(res["gt"])
        if res.get("caption_gt"):
            print("--- Caption (GT) ---")
            print(res["caption_gt"])
        print("--- Pretrained ---")
        print(res["pretrained"])
        print("--- Baseline ---")
        print(res["baseline"])
        print("--- MTL ---")
        print(res["mtl"])

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "emb_scores": emb_scores,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nSaved outputs to: {args.save_path}")


if __name__ == "__main__":
    main()
