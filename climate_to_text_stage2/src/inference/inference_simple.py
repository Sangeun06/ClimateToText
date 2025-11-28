"""
Simple inference for one Stage2 model (Stage1 encoder 기반).
- Input: Gemini JSON (global_summary + elements.image_rel_path + caption)
- We select a few samples, run Stage1 encoder + Stage2 (LoRA) → text.
- Output: Global Summary (GT), Caption (GT), Generated 텍스트.
"""

import argparse
import torch
from src.data.dataset import default_image_loader
from src.data.data_utils import build_image_transform, select_samples_gemini
from src.models.model_utils import load_stage2_lora


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple inference for one Stage2 model (Stage1 encoder 기반)")
    p.add_argument("--json-path", type=str, required=True, help="Gemini 형식 JSON (global_summary, elements)")
    p.add_argument("--image-root", type=str, required=True, help="elements.image_rel_path 기준 루트")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--prompt", type=str, default="Describe the weather situation shown in this meteorological image.")

    p.add_argument("--checkpoint-dir", type=str, required=True, help="Stage2 ckpt (LoRA) 디렉토리")
    p.add_argument("--base-model-path", type=str, required=True, help="pretrained LLM 경로 (LoRA base)")
    p.add_argument("--stage1-encoder-ckpt", type=str, required=True)
    p.add_argument("--stage1-classifier-ckpt", type=str, required=True)

    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = select_samples_gemini(args.json_path, args.image_root, args.num_samples)
    print(f"Selected {len(samples)} samples from {args.json_path}")
    if not samples:
        return

    transform = build_image_transform(args.image_size, imagenet_norm=False)

    model, tokenizer = load_stage2_lora(
        checkpoint_dir=args.checkpoint_dir,
        base_model_path=args.base_model_path,
        encoder_ckpt=args.stage1_encoder_ckpt,
        classifier_ckpt=args.stage1_classifier_ckpt,
        device=device,
        num_image_tokens=4,
    )

    for i, s in enumerate(samples, 1):
        img = default_image_loader(s["image"])
        pv = transform(img).unsqueeze(0).to(device)
        prompt = f"Condition: {s['cond_name']}\n" + args.prompt

        gen = generate_text(
            model,
            tokenizer,
            pixel_values=pv,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\n[Sample {i}] {s['image']} | cond_name={s['cond_name']}")
        print("--- Global Summary (GT) ---")
        print(s["summary"])
        if s.get("caption"):
            print("--- Caption (GT) ---")
            print(s["caption"])
        print("--- Generated ---")
        print(gen)


if __name__ == "__main__":
    main()
