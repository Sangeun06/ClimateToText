"""
Simple inference script for **pretrained-only** Mistral (text-only baseline).

- Uses the same Gemini JSON + sampling logic as `inference_simple.py`,
  but **does not use Stage2 LoRA or Stage1 encoders**.
- Instead, it feeds a text prompt to the pretrained Mistral LLM that
  includes the image condition name (`cond_name`) and a generic
  instruction like:

    "Condition: SHR6. Describe the weather situation shown in this meteorological image."

This can be used as a baseline to compare against Stage2 image-conditioned models.
"""

from __future__ import annotations

import argparse
import torch
from src.data.data_utils import select_samples_gemini
from src.models.model_utils import load_pretrained_llm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple inference for pretrained Mistral (text-only baseline).")
    p.add_argument("--json-path", type=str, required=True, help="Gemini-style JSON (global_summary, elements)")
    p.add_argument("--image-root", type=str, required=True, help="Root path to resolve elements.image_rel_path")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument(
        "--prompt",
        type=str,
        default="Describe the weather situation shown in this meteorological image.",
        help="Base instruction prompt for the LLM.",
    )
    p.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="Path to pretrained Mistral (e.g., /home/.../mistral-7b)",
    )
    return p.parse_args()


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

    samples = select_samples_gemini(args.json_path, args.image_root, args.num_samples)
    print(f"Selected {len(samples)} samples from {args.json_path}")
    if not samples:
        return

    model, tokenizer = load_pretrained_llm(args.base_model_path, device=device)

    for i, s in enumerate(samples, 1):
        # Pretrained model has no vision encoder; we only provide a text prompt.
        # We include the image condition name to give some minimal context.
        cond = s["cond_name"]
        prompt = f"Condition: {cond}.\n{args.prompt}"

        gen = generate_pretrained_text(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
        )

        print(f"\n[Sample {i}] {s['image']} | cond_name={cond}")
        print("--- Global Summary (GT) ---")
        print(s["summary"])
        if s.get("caption"):
            print("--- Caption (GT) ---")
            print(s["caption"])
        print("--- Generated (pretrained-only) ---")
        print(gen)


if __name__ == "__main__":
    main()
