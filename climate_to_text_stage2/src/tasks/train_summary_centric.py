"""
Summary-centric training:
 - 최종 목표가 이미지 -> 글로벌 서머리이므로 summary 태스크 비중을 높이고,
   cond_name(또는 type_name)을 프롬프트에 명시적으로 넣어 준다.
 - Gemini JSON만 사용 (global_summary + elements.image_rel_path + cond/type_name [+ caption]).
 - Warmup 구간에서는 caption/cot 중심, 메인 구간에서는 summary 확률을 높여 샘플링.
 - 선택적으로 caption을 summary 프롬프트 힌트로 포함 (--use-caption-hint).
 - LoRA/Stage1 encoder/DDP/wandb 지원.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from src.data.dataset import default_image_loader
from src.models.stage1_stage2_integration import Stage1ImageEncoderForStage2, ImageToTextModelStage1
from src.data.mtl_datasets import ElementCaptionDataset, ElementSummaryLinkDataset, CaseSummaryFromCaptionsDataset

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except ImportError:  # pragma: no cover
    dist = None
    DDP = None


# Summary 전용 Dataset (cond_name + optional caption_hint 포함)
class SummaryCondDataset(Dataset):
    def __init__(self, json_paths: List[str], image_root: str, include_caption: bool = False) -> None:
        super().__init__()
        self.samples: List[Dict] = []
        root = Path(os.path.expanduser(image_root)).resolve()
        for jp in json_paths:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            for case in data:
                summary = (case.get("global_summary") or "").strip()
                if not summary:
                    continue
                for el in case.get("elements", []):
                    if el.get("error") is not None:
                        continue
                    rel = el.get("image_rel_path") or ""
                    cond = el.get("cond_name") or el.get("type_name") or ""
                    cap = (el.get("caption") or "").strip()
                    if not rel or not cond:
                        continue
                    rel_clean = rel.lstrip("./")
                    img_path = root / rel_clean
                    # WeatherQA_MD_* 하위도 탐색
                    if not img_path.exists():
                        for sub in ["WeatherQA_MD_2014-2019", "WeatherQA_MD_2020"]:
                            cand = root / sub / rel_clean
                            if cand.exists():
                                img_path = cand
                                break
                    if not img_path.exists():
                        continue
                    self.samples.append(
                        {
                            "image": str(img_path),
                            "cond_name": cond,
                            "summary": summary,
                            "caption": cap if include_caption else "",
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def collate_summary(batch, tokenizer, image_tf, max_seq_len: int, prompt_template: str):
    pvs, ids_list, attn_list, labels_list = [], [], [], []
    for b in batch:
        img = default_image_loader(b["image"])
        pv = image_tf(img)
        pvs.append(pv)
        prompt = prompt_template.format(cond_name=b["cond_name"], caption_hint=b.get("caption", ""))
        target = b["summary"]
        p_enc = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_seq_len)
        t_enc = tokenizer(target, add_special_tokens=False, truncation=True, max_length=max_seq_len)
        ids = p_enc["input_ids"] + t_enc["input_ids"]
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[: len(p_enc["input_ids"])] = -100
        ids_list.append(input_ids)
        attn_list.append(attn)
        labels_list.append(labels)
    input_ids = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    pvs = torch.stack(pvs, dim=0)
    return {"pixel_values": pvs, "input_ids": input_ids, "attention_mask": attn, "labels": labels}


def collate_mtl(batch, tokenizer, image_tf, max_seq_len: int):
    # caption/cot_link 공용 collate (datasets.py와 동일한 방식)
    pvs, ids_list, attn_list, labels_list, tasks = [], [], [], [], []
    for sample in batch:
        tasks.append(sample.task)
        if sample.image_path is not None and os.path.exists(sample.image_path):
            img = default_image_loader(sample.image_path)
            pv = image_tf(img)
        else:
            pv = torch.zeros(3, image_tf.transforms[0].size[0], image_tf.transforms[0].size[1])
        # prompt/target 토큰화 (target이 짤리지 않도록 prompt 길이를 줄인다)
        prompt_ids = tokenizer(
            sample.prompt, add_special_tokens=True, truncation=True, max_length=max_seq_len
        )["input_ids"]
        target_ids = tokenizer(
            sample.target, add_special_tokens=False, truncation=True, max_length=max_seq_len
        )["input_ids"]
        if len(target_ids) == 0:
            target_ids = [tokenizer.eos_token_id]
        min_target = min(len(target_ids), max_seq_len // 2)
        allow_prompt = max_seq_len - min_target
        prompt_ids = prompt_ids[:allow_prompt]
        ids = prompt_ids + target_ids
        ids = ids[:max_seq_len]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100
        ids_list.append(input_ids)
        attn_list.append(attn)
        labels_list.append(labels)
        pvs.append(pv)
    input_ids = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    pvs = torch.stack(pvs, dim=0)
    return {"pixel_values": pvs, "input_ids": input_ids, "attention_mask": attn, "labels": labels, "tasks": tasks}


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    seq_len = min(shift_logits.size(1), shift_labels.size(1))
    shift_logits = shift_logits[:, :seq_len, :]
    shift_labels = shift_labels[:, :seq_len]
    vocab = shift_logits.size(-1)
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())
    mask = shift_labels != -100
    token_per_sample = mask.sum(dim=1).clamp(min=1)
    loss_per_sample = (loss_flat * mask).sum(dim=1) / token_per_sample
    return loss_per_sample.mean()


def decode_labels(labels: torch.Tensor, tokenizer) -> str:
    if labels.ndim > 1:
        labels = labels[0]
    arr = labels.clone()
    arr[arr == -100] = tokenizer.pad_token_id
    return tokenizer.decode(arr, skip_special_tokens=True).strip()


def parse_args():
    p = argparse.ArgumentParser(description="Summary-centric MTL (image -> global summary 우선)")
    p.add_argument("--gemini-json", nargs="+", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--model-path", default="/home/agi592/models/mistral-7b")
    p.add_argument("--output-dir", default="twkim/checkpoints/stage2_summary_centric")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup-epochs", type=int, default=0, help="caption/cot 위주 워밍업 epoch 수")
    p.add_argument("--main-epochs", type=int, default=1, help="summary 우선 학습 epoch 수")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--encoder-lr", type=float, default=None)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--train-val-split", type=float, default=0.95)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--use-caption-hint", action="store_true")

    # 태스크 샘플링 확률 (main 단계)
    p.add_argument("--prob-summary", type=float, default=0.6)
    p.add_argument("--prob-caption", type=float, default=0.2)
    p.add_argument("--prob-cot", type=float, default=0.2)
    p.add_argument("--prob-text-summary", type=float, default=0.0, help="Task D (captions -> summary) 확률")
    p.add_argument("--sample-log-steps", type=int, default=500, help=">0이면 지정 스텝마다 샘플 생성 결과를 wandb에 기록")
    p.add_argument("--sample-log-max-new-tokens", type=int, default=80)
    p.add_argument("--force-fp32", action="store_true", help="강제로 float32로 로드/학습 (AMP 끄기)")

    # LoRA
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--num-image-tokens", type=int, default=4)

    # Stage1
    p.add_argument("--stage1-encoder-ckpt", required=True)
    p.add_argument("--stage1-classifier-ckpt", required=True)
    p.add_argument("--stage1-trainable-blocks", type=int, default=0)

    # wandb
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="climate-to-text-stage2-summary-centric")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-api-key", type=str, default="")

    # 분산
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--local-rank", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    use_ddp = args.distributed or "LOCAL_RANK" in os.environ
    if use_ddp and (dist is None or DDP is None):
        raise ImportError("torch.distributed 필요. --distributed 끄거나 torchrun 사용")
    if use_ddp:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
    if rank == 0:
        print(f"Using device: {device}, distributed={use_ddp}, world_size={world_size}")

    # wandb
    wandb = None
    if args.use_wandb and rank == 0:
        try:
            import wandb as _wandb

            wandb = _wandb
            if args.wandb_api_key:
                wandb.login(key=args.wandb_api_key, relogin=True)
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or None,
                entity=args.wandb_entity or None,
                config=vars(args),
            )
        except Exception as e:
            print(f"wandb init 실패: {e}")
            wandb = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.force_fp32 or not torch.cuda.is_available():
        load_dtype = torch.float32
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float16

    llm = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=load_dtype, device_map=None)
    llm.resize_token_embeddings(len(tokenizer))
    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_cfg)
        llm.print_trainable_parameters()
    llm.to(device)

    # Datasets
    summary_ds = SummaryCondDataset(args.gemini_json, args.image_root, include_caption=args.use_caption_hint)
    # 데이터 없는 경우 바로 중단
    if len(summary_ds) == 0:
        raise ValueError("No usable summary samples found. Check gemini-json / image-root / cond_name/type_name.")
    caption_ds = ElementCaptionDataset(args.gemini_json[0], args.image_root)  # concat handled below
    for jp in args.gemini_json[1:]:
        caption_ds.samples.extend(ElementCaptionDataset(jp, args.image_root).samples)
    cot_ds = ElementSummaryLinkDataset(args.gemini_json[0], args.image_root)
    for jp in args.gemini_json[1:]:
        cot_ds.samples.extend(ElementSummaryLinkDataset(jp, args.image_root).samples)
    text_summary_ds = CaseSummaryFromCaptionsDataset(args.gemini_json[0], args.image_root)
    for jp in args.gemini_json[1:]:
        text_summary_ds.samples.extend(CaseSummaryFromCaptionsDataset(jp, args.image_root).samples)

    if rank == 0:
        print(
            f"Dataset sizes | summary: {len(summary_ds)}, caption: {len(caption_ds.samples)}, cot: {len(cot_ds.samples)}"
        )

    # train/val split for summary only (메인 목표), caption/cot는 전체 사용
    # rank 0에서 split 후 broadcast하여 모든 rank가 동일 split 사용
    total_len = len(summary_ds)
    if rank == 0:
        eval_len = int(total_len * (0.2))  # 20%를 eval로 떼어냄
        remain_len = total_len - eval_len
        summary_remain, summary_eval = torch.utils.data.random_split(summary_ds, [remain_len, eval_len])
        indices_eval = summary_eval.indices

        # 평가용(evaluation) 데이터셋 기록
        eval_samples = [summary_ds[i] for i in indices_eval]
        with open(os.path.join(args.output_dir, "eval_summary_samples.json"), "w", encoding="utf-8") as f:
            json.dump(eval_samples, f, ensure_ascii=False, indent=2)
            print(f"평가용 데이터셋을 {f.name}에 저장했습니다.")

        # 나머지에서 train/val split
        train_len = int(remain_len * args.train_val_split)
        val_len = remain_len - train_len
        summary_train, summary_val = torch.utils.data.random_split(summary_remain, [train_len, val_len])
        indices_train = summary_train.indices
        indices_val = summary_val.indices
        
        # 검증용 (validation) 데이터셋 기록
        val_samples = [summary_ds[i] for i in indices_val]
        with open(os.path.join(args.output_dir, "val_summary_samples.json"), "w", encoding="utf-8") as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)
    else:
        indices_train = indices_val = None

    if use_ddp:
        obj = [indices_train, indices_val]
        dist.broadcast_object_list(obj, src=0)
        indices_train, indices_val = obj

    summary_train = torch.utils.data.Subset(summary_ds, indices_train)
    summary_val = torch.utils.data.Subset(summary_ds, indices_val)

    sampler_summary_train = DistributedSampler(summary_train, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    sampler_summary_val = DistributedSampler(summary_val, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    sampler_caption = DistributedSampler(caption_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    sampler_cot = DistributedSampler(cot_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    sampler_text = DistributedSampler(text_summary_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None

    image_tf = build_transform(args.image_size)
    prompt_template = (
        "You are a meteorologist at SPC.\n"
        "Condition: {cond_name}\n"
        + ("Element hint: {caption_hint}\n" if args.use_caption_hint else "")
        + "Given this mesoscale diagnostic image, write a concise severe weather SUMMARY (2–3 sentences).\n"
        "Must clearly state hazards (hail/wind/tornado), affected regions, timing/evolution, and whether a watch is needed.\n\n"
        "Summary:"
    )
    collate_sum = lambda b: collate_summary(b, tokenizer, image_tf, args.max_seq_len, prompt_template)

    summary_train_loader = DataLoader(
        summary_train,
        batch_size=args.batch_size,
        shuffle=(sampler_summary_train is None),
        sampler=sampler_summary_train,
        num_workers=args.num_workers,
        collate_fn=collate_sum,
    )
    summary_val_loader = DataLoader(
        summary_val,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler_summary_val,
        num_workers=args.num_workers,
        collate_fn=collate_sum,
    )
    caption_loader = DataLoader(
        caption_ds,
        batch_size=args.batch_size,
        shuffle=(sampler_caption is None),
        sampler=sampler_caption,
        num_workers=args.num_workers,
        collate_fn=(lambda b: collate_mtl(b, tokenizer, image_tf, args.max_seq_len)),
    )
    cot_loader = DataLoader(
        cot_ds,
        batch_size=args.batch_size,
        shuffle=(sampler_cot is None),
        sampler=sampler_cot,
        num_workers=args.num_workers,
        collate_fn=(lambda b: collate_mtl(b, tokenizer, image_tf, args.max_seq_len)),
    )
    text_loader = DataLoader(
        text_summary_ds,
        batch_size=args.batch_size,
        shuffle=(sampler_text is None),
        sampler=sampler_text,
        num_workers=args.num_workers,
        collate_fn=(lambda b: collate_mtl(b, tokenizer, image_tf, args.max_seq_len)),
    )

    # Stage1 encoder + model
    image_encoder = Stage1ImageEncoderForStage2(
        encoder_ckpt_path=args.stage1_encoder_ckpt,
        classifier_ckpt_path=args.stage1_classifier_ckpt,
        device=device,
        freeze=(args.stage1_trainable_blocks <= 0),
        num_trainable_blocks=max(0, args.stage1_trainable_blocks),
    )
    model = ImageToTextModelStage1(
        llm=llm,
        image_encoder=image_encoder,
        num_image_tokens=args.num_image_tokens,
    ).to(device)
    if use_ddp:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    # Optimizer/scheduler
    lr_enc = args.encoder_lr if args.encoder_lr is not None else args.lr
    enc_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if "image_encoder" in name else other_params).append(p)
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr})
    if enc_params:
        param_groups.append({"params": enc_params, "lr": lr_enc})
    optimizer = AdamW(param_groups)

    total_steps = (len(summary_train_loader) * args.main_epochs) + (len(caption_loader) + len(cot_loader)) * args.warmup_epochs
    warmup_steps = int(total_steps * args.warmup_ratio) if total_steps > 0 else 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_steps, 1)
    )
    use_amp = device.type == "cuda" and load_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Iterators for sampling
    def reset_iter(loader):
        it = iter(loader)
        return it

    cap_it, cot_it, text_it, sum_it = None, None, None, None

    global_step = 0
    best_val = float("inf")

    def train_step(batch):
        nonlocal global_step
        optimizer.zero_grad()
        pv = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # 타깃 토큰이 없으면 스킵 (loss=0 기록 방지)
        if not torch.any(labels != -100):
            return None
        if use_amp:
            with torch.cuda.amp.autocast(dtype=load_dtype):
                out = model(
                    pixel_values=pv,
                    cond_ids=None,
                    input_ids=ids,
                    attention_mask=attn,
                    labels=labels,
                )
                loss = out["loss"]
        else:
            out = model(
                pixel_values=pv,
                cond_ids=None,
                input_ids=ids,
                attention_mask=attn,
                labels=labels,
            )
            loss = out["loss"]
        if not torch.isfinite(loss):
            return None
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        global_step += 1
        return loss

    # Warmup: caption + cot
    for ep in range(args.warmup_epochs):
        if use_ddp and sampler_caption is not None:
            sampler_caption.set_epoch(ep)
        if use_ddp and sampler_cot is not None:
            sampler_cot.set_epoch(ep)
        cap_it, cot_it = reset_iter(caption_loader), reset_iter(cot_loader)
        loaders = [("caption", cap_it, caption_loader), ("cot", cot_it, cot_loader)]
        for name, it, loader in loaders:
            if len(loader) == 0:
                continue
            for _ in range(len(loader)):
                try:
                    batch = next(it)
                except StopIteration:
                    it = reset_iter(loader)
                    batch = next(it)
                loss = train_step(batch)
                if loss is None:
                    continue
                if rank == 0 and args.use_wandb and wandb is not None:
                    wandb.log({f"train/{name}_loss_step": loss.item()}, step=global_step)
                if rank == 0 and args.log_every > 0 and global_step % args.log_every == 0:
                    print(f"[warmup ep{ep+1} step{global_step}] {name} loss={loss.item():.4f}")

    # Main: summary 우선, caption/cot 보조 (확률 샘플링)
    for ep in range(args.main_epochs):
        if use_ddp and sampler_summary_train is not None:
            sampler_summary_train.set_epoch(ep)
        if use_ddp and sampler_caption is not None:
            sampler_caption.set_epoch(args.warmup_epochs + ep)
        if use_ddp and sampler_cot is not None:
            sampler_cot.set_epoch(args.warmup_epochs + ep)
        if use_ddp and sampler_text is not None:
            sampler_text.set_epoch(args.warmup_epochs + ep)
        sum_it, cap_it, cot_it, text_it = (
            reset_iter(summary_train_loader),
            reset_iter(caption_loader),
            reset_iter(cot_loader),
            reset_iter(text_loader),
        )
        probs = [
            ("summary", args.prob_summary, sum_it, summary_train_loader),
            ("caption", args.prob_caption, cap_it, caption_loader),
            ("cot", args.prob_cot, cot_it, cot_loader),
        ]
        if args.prob_text_summary > 0:
            probs.append(("text_summary", args.prob_text_summary, text_it, text_loader))
        total_prob = sum(p for _, p, _, _ in probs)
        probs = [(n, p / total_prob, it, ld) for n, p, it, ld in probs]

        steps_in_epoch = len(summary_train_loader)  # 기준: summary 스텝 수
        for _ in range(steps_in_epoch):
            r = random.random()
            acc = 0.0
            chosen = probs[-1]
            for cand in probs:
                acc += cand[1]
                if r <= acc:
                    chosen = cand
                    break
            name, _, it, loader = chosen
            try:
                batch = next(it)
            except StopIteration:
                it = reset_iter(loader)
                batch = next(it)
                # update iterator reference
                for idx, (n, p, _, ld) in enumerate(probs):
                    if n == name:
                        probs[idx] = (n, p, it, ld)
            loss = train_step(batch)
            if loss is None:
                continue
            if rank == 0 and args.use_wandb and wandb is not None:
                wandb.log({f"train/{name}_loss_step": loss.item()}, step=global_step)
            if rank == 0 and args.log_every > 0 and global_step % args.log_every == 0:
                print(f"[main ep{ep+1} step{global_step}] {name} loss={loss.item():.4f}")
            if (
                args.sample_log_steps
                and args.sample_log_steps > 0
                and rank == 0
                and args.use_wandb
                and wandb is not None
                and global_step % args.sample_log_steps == 0
            ):
                model_eval = model.module if use_ddp else model
                model_eval.eval()
                with torch.no_grad():
                    pv_sample = batch["pixel_values"][:1].to(device)
                    gen_text = model_eval.generate(
                        pixel_values=pv_sample,
                        cond_ids=None,
                        tokenizer=tokenizer,
                        generation_prompt="",
                        max_new_tokens=args.sample_log_max_new_tokens,
                        num_beams=1,
                        no_repeat_ngram_size=4,
                        repetition_penalty=1.1,
                    )[0]
                    labels_sample = batch["labels"][:1].to(device)
                    gt_text = decode_labels(labels_sample, tokenizer)
                wandb.log(
                    {
                        "sample/global_step": global_step,
                        "sample/generated": gen_text,
                        "sample/gt": gt_text,
                    },
                    step=global_step,
                )
                model_eval.train()

        # Epoch-end train loss는 스텝 평균 대신 wandb/콘솔에 summary 위주로만 보고 싶다면 생략 가능
        if rank == 0:
            # Validation (summary only)
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for batch in summary_val_loader:
                    pv = batch["pixel_values"].to(device)
                    ids = batch["input_ids"].to(device)
                    attn = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    out = model(
                        pixel_values=pv,
                        cond_ids=None,
                        input_ids=ids,
                        attention_mask=attn,
                        labels=labels,
                    )
                    loss = out["loss"]
                    total_val += loss.item()
            avg_val = total_val / max(len(summary_val_loader), 1)
            print(f"[main ep{ep+1}] Val(summary) loss: {avg_val:.4f}")
            if args.use_wandb and wandb is not None:
                wandb.log({f"val/summary_loss": avg_val}, step=global_step)
            if avg_val < best_val:
                best_val = avg_val
                print(f"New best val {best_val:.4f}, saving checkpoint...")
                # save
                os.makedirs(args.output_dir, exist_ok=True)
                (model.module if use_ddp else model).llm.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                torch.save(
                    {
                        "class_embeddings": (model.module if use_ddp else model).class_embeddings.state_dict(),
                        "image_projection": (model.module if use_ddp else model).image_projection.state_dict(),
                    },
                    os.path.join(args.output_dir, "stage2_prefix_modules.pt"),
                )
                enc = getattr(model.module if use_ddp else model, "image_encoder", None)
                if enc is not None and hasattr(enc, "pipeline") and hasattr(enc.pipeline, "encoder"):
                    try:
                        enc_state = enc.pipeline.encoder.state_dict()
                        torch.save(enc_state, os.path.join(args.output_dir, "stage1_encoder_finetuned.pt"))
                    except Exception:
                        pass
            model.train()

    if rank == 0:
        print("Training finished.")
        if args.use_wandb and wandb is not None:
            wandb.finish()
    if use_ddp and dist is not None and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
