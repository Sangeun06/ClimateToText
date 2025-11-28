"""
Multi-task Stage2 training script for WeatherQA + Gemini captions.

태스크 구성:
  - Task A: 요소 이미지 → Gemini caption (4문장) 재구성
  - Task B: MD-style summary (이미지 → 요약) 생성
  - Task C: 요소 caption + global summary → 연결/설명 한두 문장
  - Task D (옵션): 여러 caption → global summary (텍스트 기반 요약)

공통:
  - Stage1 patch-perceiver encoder + EfficientNet classifier로부터 이미지 latent와 cond id를 얻고,
  - 이를 LLM(Mistral-7B) prefix 토큰으로 주입해 텍스트를 생성하도록 LoRA로 미세조정한다.

사용 예시는 README.md 참고.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from src.models.stage1_stage2_integration import Stage1ImageEncoderForStage2, ImageToTextModelStage1
from src.data.mtl_datasets import build_mtl_dataset_and_loader

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except ImportError:  # pragma: no cover
    dist = None
    DDP = None

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-task Stage2 training for WeatherQA + Gemini captions")

    # 데이터 경로
    p.add_argument(
        "--gemini-json",
        type=str,
        nargs="+",
        required=True,
        help="WeatherQA Gemini batch JSON 경로들 (여러 개 지정 가능)",
    )
    p.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="elements.image_rel_path 기준 루트 디렉토리 (예: data/WeatherQA)",
    )

    # 모델/출력 경로
    p.add_argument(
        "--model-path",
        type=str,
        default="/home/agi592/models/mistral-7b",
        help="사전학습 LLM 경로",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="twkim/checkpoints/stage2_mtl_weatherqa",
        help="체크포인트 저장 경로",
    )

    # 하이퍼파라미터
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5, help="LLM/Stage2 부분 학습률")
    p.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help="Stage1 encoder 뒷단 학습률 (None이면 --lr과 동일, Stage1 완전 freeze 시에는 무시)",
    )
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-seq-len", type=int, default=768)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--train-val-split", type=float, default=0.95)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=200, help="step 단위 로깅 주기 (loss 출력)")

    # LoRA 설정
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--num-image-tokens", type=int, default=4)
    p.add_argument("--force-fp32", action="store_true", help="강제로 float32로 로드/학습 (AMP 끄기)")
    p.add_argument("--sample-log-steps", type=int, default=500, help=">0이면 지정 스텝마다 샘플 생성 결과를 wandb에 기록")
    p.add_argument("--sample-log-max-new-tokens", type=int, default=80)

    # Stage1 encoder 설정
    p.add_argument(
        "--stage1-encoder-ckpt",
        type=str,
        required=True,
        help="Stage1 encoder checkpoint 경로 (stage1_vision_encoder_mae.pt 등)",
    )
    p.add_argument(
        "--stage1-classifier-ckpt",
        type=str,
        required=True,
        help="Stage1 standalone classifier checkpoint 경로",
    )
    p.add_argument(
        "--stage1-trainable-blocks",
        type=int,
        default=0,
        help="Stage1 encoder 마지막 몇 개 block을 finetune할지 (0이면 완전 freeze)",
    )

    # 태스크별 loss 가중치
    # 단순화를 위해 태스크 가중치는 제거

    # 태스크 포함 여부 (원하면 일부만 켜고 끌 수 있음)
    p.add_argument("--no-caption", action="store_true", help="Task A (caption) 비활성화")
    p.add_argument("--no-summary", action="store_true", help="Task B (summary) 비활성화")
    p.add_argument("--no-cot-link", action="store_true", help="Task C (cot_link) 비활성화")
    p.add_argument("--text-summary", action="store_true", help="Task D (text_summary) 활성화")

    # wandb
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="climate-to-text-stage2-mtl")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-api-key", type=str, default="")

    # 분산 학습 옵션
    p.add_argument("--distributed", action="store_true", help="torchrun/torch.distributed 기반 DDP 사용")
    p.add_argument("--local-rank", type=int, default=0, help="torchrun에서 전달되는 local rank")

    return p.parse_args()


def save_checkpoint(
    model: ImageToTextModelStage1,
    tokenizer,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model.llm.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(
        {
            "class_embeddings": model.class_embeddings.state_dict(),
            "image_projection": model.image_projection.state_dict(),
        },
        os.path.join(output_dir, "stage2_prefix_modules.pt"),
    )

    # Stage1 encoder를 finetune했다면 encoder 가중치도 저장
    enc = getattr(model, "image_encoder", None)
    if enc is not None and hasattr(enc, "pipeline") and hasattr(enc.pipeline, "encoder"):
        try:
            enc_state = enc.pipeline.encoder.state_dict()
        except Exception:
            enc_state = None
        else:
            torch.save(
                enc_state,
                os.path.join(output_dir, "stage1_encoder_finetuned.pt"),
            )

    print(f"Checkpoint saved to {output_dir}")


def compute_normalized_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Shifted cross-entropy를 토큰 수로 나눠 sample 평균 후 배치 평균.
    길이 불일치 시 공통 길이로 잘라 안전하게 계산.
    """
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


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 분산 초기화
    use_ddp = args.distributed or "LOCAL_RANK" in os.environ
    if use_ddp and (dist is None or DDP is None):
        raise ImportError("torch.distributed/torch.nn.parallel가 필요합니다. 분산 옵션을 끄거나 torchrun 환경을 사용하세요.")

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

    # wandb 초기화 (rank 0에서만)
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

    # 1) 토크나이저 및 LLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.force_fp32 or not torch.cuda.is_available():
        load_dtype = torch.float32
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float16

    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=load_dtype,
        device_map=None,
    )
    llm.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        print("Applying LoRA adapters to LLM...")
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

    # 2) DataLoader (multi-task)
    include_caption = not args.no_caption
    include_summary = not args.no_summary
    include_cot_link = not args.no_cot_link
    include_text_summary = args.text_summary

    train_loader, val_loader, train_sampler, val_sampler = build_mtl_dataset_and_loader(
        gemini_jsons=args.gemini_json,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_size=args.image_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        train_val_split=args.train_val_split,
        num_workers=args.num_workers,
        include_caption=include_caption,
        include_summary=include_summary,
        include_cot_link=include_cot_link,
        include_text_summary=include_text_summary,
        distributed=use_ddp,
        world_size=world_size,
        rank=rank,
    )

    # 3) Stage1 encoder + Stage2 모델 구성
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
    )
    model.to(device)

    # 4) Optimizer (encoder lr 분리)
    lr_enc = args.encoder_lr if args.encoder_lr is not None else args.lr
    enc_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "image_encoder" in name:
            enc_params.append(p)
        else:
            other_params.append(p)

    param_groups: List[Dict] = []
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr})
    if enc_params:
        param_groups.append({"params": enc_params, "lr": lr_enc})

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable params: {sum(p.numel() for p in trainable_params):,} "
        f"(encoder lr={lr_enc}, others lr={args.lr})"
    )

    if use_ddp:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)

    optimizer = AdamW(param_groups)

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda" and load_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0
        num_train_steps = 0
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # rank 0에서만 tqdm 사용 (DDP 시 다른 rank는 조용히 학습)
        iterator = train_loader
        if rank == 0:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [train]")
            except Exception:
                iterator = train_loader

        for batch in iterator:
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tasks = batch.get("tasks", [])

            # 타깃 토큰이 하나도 없으면 학습 스킵 (loss=0 기록 방지)
            if not torch.any(labels != -100):
                continue

            if use_amp:
                with torch.cuda.amp.autocast(dtype=load_dtype):
                    out = model(
                        pixel_values=pixel_values,
                        cond_ids=None,  # Stage1 classifier 내부에서 cond id 사용
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = out["loss"]
            else:
                out = model(
                    pixel_values=pixel_values,
                    cond_ids=None,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                # normalized CE
                logits = out["logits"]
                loss = compute_normalized_loss(logits, labels)

            if not torch.isfinite(loss):
                print(f"Non-finite loss {loss.item()} at step {global_step}, skip.")
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            scheduler.step()

            total_train_loss += loss.item()
            num_train_steps += 1
            global_step += 1

            if rank == 0 and args.use_wandb and wandb is not None:
                # 태스크별 손실도 가능하면 분리해서 기록 (batch 내 태스크가 단일일 때)
                task_logs = {}
                tasks = batch.get("tasks", None)
                if tasks and len(tasks) > 0:
                    first = tasks[0]
                    if all(t == first for t in tasks):
                        task_logs[f"train/{first}_loss_step"] = loss.item()
                wandb.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )
                if task_logs:
                    wandb.log(task_logs, step=global_step)
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
                    pv_sample = pixel_values[:1]
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
                    gt_text = decode_labels(labels[:1], tokenizer)
                wandb.log(
                    {
                        "sample/global_step": global_step,
                        "sample/generated": gen_text,
                        "sample/gt": gt_text,
                    },
                    step=global_step,
                )
                model_eval.train()

            # 콘솔 로그
            if rank == 0 and args.log_every > 0 and global_step % args.log_every == 0:
                print(f"[Epoch {epoch+1} | Step {global_step}] loss={loss.item():.4f}")

        avg_train_loss = total_train_loss / max(num_train_steps, 1)
        print(f"[Epoch {epoch+1}] Train loss: {avg_train_loss:.4f}")
        if rank == 0 and args.use_wandb and wandb is not None:
            wandb.log(
                {"train/loss_epoch": avg_train_loss, "train/epoch": epoch + 1},
                step=global_step,
            )

        # Validation
        # Validation (rank 0만 수행)
        if rank == 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    out = model(
                        pixel_values=pixel_values,
                        cond_ids=None,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    logits = out["logits"]
                    loss = compute_normalized_loss(logits, labels)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / max(len(val_loader), 1)
            print(f"[Epoch {epoch+1}] Val loss: {avg_val_loss:.4f}")
            if args.use_wandb and wandb is not None:
                wandb.log(
                    {"val/loss": avg_val_loss, "val/epoch": epoch + 1},
                    step=global_step,
                )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best val loss {best_val_loss:.4f}, saving checkpoint...")
                save_checkpoint(model.module if use_ddp else model, tokenizer, args.output_dir)

    if rank == 0:
        print("Training finished.")
        if args.use_wandb and wandb is not None:
            wandb.finish()

    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
