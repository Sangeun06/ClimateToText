"""
Quick cot_link sanity checker.

실행하면 cot_link 샘플을 현재 프롬프트/토크나이저 설정으로 토큰화해서
 - 총 샘플 수
 - 토큰화 전에 target 비어 있는 샘플 수
 - prompt+target 합친 뒤에도 유효 토큰(-100이 아닌)이 0개인 샘플 수
를 출력합니다.

기본 경로는 현재 학습에 쓰는 Gemini JSON / Mistral 토크나이저 기준입니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

from src.data.mtl_datasets import _split_sentences


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--json-paths",
        nargs="+",
        default=[
            "kse/ClimateToText/data/WeatherQA/gemini_element_captions_2014-2019.json",
            "kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json",
        ],
        help="Gemini caption JSON 경로들",
    )
    p.add_argument(
        "--tokenizer-path",
        default="/home/agi592/models/mistral-7b",
        help="토크나이저 경로",
    )
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--print-samples", type=int, default=0, help="앞부분 몇 개 샘플(prompt/target)을 출력")
    p.add_argument("--target-len-threshold", type=int, default=4, help="이 길이 이하인 target을 따로 출력")
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tok.pad_token = tok.eos_token

    total = empty_before = all_mask_after = 0
    prompt_lens: List[int] = []
    target_lens: List[int] = []

    printed = 0
    short_samples: List[tuple] = []  # (prompt, target, plen, tlen)

    for jp in args.json_paths:
        data = json.load(open(jp, "r", encoding="utf-8"))
        for case in data:
            gs = (case.get("global_summary") or "").strip()
            if not gs:
                continue
            for el in case.get("elements", []):
                if el.get("error") is not None:
                    continue
                cap = (el.get("caption") or "").strip()
                if not cap:
                    continue

                # prompt/target (현재 cot_link 프롬프트와 동일)
                prompt = f"{cap}\n\nForecast summary:\n{gs}\n\nExplain the link:\n"
                sentences = _split_sentences(cap)
                target = sentences[-1] if sentences else ""

                total += 1

                prompt_ids = tok(
                    prompt, add_special_tokens=True, truncation=True, max_length=args.max_seq_len
                )["input_ids"]
                target_ids = tok(
                    target, add_special_tokens=False, truncation=True, max_length=args.max_seq_len
                )["input_ids"]

                if len(target_ids) == 0:
                    empty_before += 1
                    target_ids = [tok.eos_token_id]

                min_target = min(len(target_ids), args.max_seq_len // 2)
                allow_prompt = args.max_seq_len - min_target
                prompt_ids = prompt_ids[:allow_prompt]

                ids = (prompt_ids + target_ids)[: args.max_seq_len]
                labels = torch.tensor(ids)
                labels[: len(prompt_ids)] = -100

                prompt_lens.append(len(prompt_ids))
                target_lens.append(len(target_ids))
                if not torch.any(labels != -100):
                    all_mask_after += 1

                if args.print_samples and printed < args.print_samples:
                    printed += 1
                    print("\n==== Sample", printed, "====")
                    print(f"prompt_len={len(prompt_ids)}, target_len={len(target_ids)}")
                    print("--- Prompt ---")
                    print(prompt.strip())
                    print("--- Target ---")
                    print(target.strip())

                if len(target_ids) <= args.target_len_threshold:
                    short_samples.append((prompt, target, len(prompt_ids), len(target_ids)))

    def stats(arr: List[int]):
        if not arr:
            return "n/a"
        return f"mean={sum(arr)/len(arr):.1f}, max={max(arr)}, min={min(arr)}"

    print(f"total cot samples: {total}")
    print(f"target empty before trunc: {empty_before}")
    print(f"labels all -100 after trunc: {all_mask_after}")
    if total > 0:
        print(f"ratio empty_before: {empty_before/total:.4f}")
        print(f"ratio all_-100_after: {all_mask_after/total:.4f}")
    print(f"prompt token length stats: {stats(prompt_lens)}")
    print(f"target token length stats: {stats(target_lens)}")

    if short_samples:
        print(f"\n=== Targets with length <= {args.target_len_threshold} (showing up to 10) ===")
        for i, (pr, ta, pl, tl) in enumerate(short_samples[:10], start=1):
            print(f"\n[# {i}] prompt_len={pl}, target_len={tl}")
            print("--- Prompt ---")
            print(pr.strip())
            print("--- Target ---")
            print(ta.strip())


if __name__ == "__main__":
    main()
