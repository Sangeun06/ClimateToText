"""
Shared model-loading utilities for Stage2 (LoRA + Stage1 encoder) and pretrained-only baselines.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .stage1_stage2_integration import Stage1ImageEncoderForStage2, ImageToTextModelStage1


def load_stage1_encoder(
    encoder_ckpt: str,
    classifier_ckpt: str,
    device: torch.device,
    finetuned_path: str | None = None,
    freeze: bool = True,
):
    image_encoder = Stage1ImageEncoderForStage2(
        encoder_ckpt_path=encoder_ckpt,
        classifier_ckpt_path=classifier_ckpt,
        device=device,
        freeze=freeze,
    )
    if finetuned_path and os.path.exists(finetuned_path):
        if hasattr(image_encoder, "pipeline") and hasattr(image_encoder.pipeline, "encoder"):
            try:
                enc_state = torch.load(finetuned_path, map_location=device, weights_only=True)
            except TypeError:
                enc_state = torch.load(finetuned_path, map_location=device)
            try:
                image_encoder.pipeline.encoder.load_state_dict(enc_state, strict=False)
                print(f"Loaded finetuned Stage1 encoder from {finetuned_path}")
            except Exception as e:  # pragma: no cover
                print(f"Warning: failed to load finetuned Stage1 encoder: {e}")
    return image_encoder


def load_stage2_lora(
    checkpoint_dir: str,
    base_model_path: str,
    encoder_ckpt: str,
    classifier_ckpt: str,
    device: torch.device,
    num_image_tokens: int = 4,
) -> Tuple[ImageToTextModelStage1, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    llm = PeftModel.from_pretrained(base_llm, checkpoint_dir)
    llm.to(device)

    finetuned_enc_path = os.path.join(checkpoint_dir, "stage1_encoder_finetuned.pt")
    image_encoder = load_stage1_encoder(
        encoder_ckpt=encoder_ckpt,
        classifier_ckpt=classifier_ckpt,
        device=device,
        finetuned_path=finetuned_enc_path,
        freeze=True,
    )

    model = ImageToTextModelStage1(
        llm=llm,
        image_encoder=image_encoder,
        num_image_tokens=num_image_tokens,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def load_pretrained_llm(base_model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer
