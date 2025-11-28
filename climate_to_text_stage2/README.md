# Stage2 (Image → MD‑Style Summary) with MTL

Stage2 fine‑tunes Mistral‑7B (LoRA) to generate SPC Mesoscale Discussion–style summaries from meteorological images. We reuse a Stage1 image encoder + classifier (WeatherQA) to provide:
- **Image latent** (768‑dim)
- **Image type** (`cond_name`, 20 classes)

To make the LLM reason about the images, we train it with **multi‑task learning (MTL)** on Gemini‑augmented WeatherQA data (captions + summaries).

---

## Repository Layout (key files)

- `src/`
  - `data/` — `dataset.py` (default loader), `mtl_datasets.py` (MTL tasks), `data_utils.py` (sampling/paths)
  - `models/` — `stage1_stage2_integration.py` (Stage1 wrapper), `model_utils.py` (LoRA/pretrained loaders)
  - `tasks/` — `train_mtl_weatherqa.py` (Uniform MTL), `train_summary_centric.py` (Summary‑focused MTL)
  - `inference/` — `inference_simple.py` (LoRA), `inference_simple_pretrained.py` (text‑only), `compare_inference_mtl.py`
  - `eval/` — `text_similarity_embedding.py`, `text_similarity_llm_judge.py`, `check_cot_stats.py`, (legacy) `eval_simple.py`
- `README.md` — setup, commands
- `.gitignore`, `requirements.txt`

**Dataset expected**: Gemini JSON, e.g.  
`kse/ClimateToText/data/WeatherQA/gemini_element_captions_2014-2019.json`  
`kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json`

Each case contains:
```json
{
  "global_summary": "...",
  "elements": [
    {
      "image_rel_path": ".../md_image/...gif",
      "cond_name": "shr6",
      "caption": "Sentence1. Sentence2. Sentence3. Sentence4."
    },
    ...
  ]
}
```
Caption = 4 sentences (visual description ×2, physical meaning ×1, link to summary ×1).

---

## Setup

0) Create/activate Conda env (example name: `climate-stage2`)
```
conda create -n climate-stage2 python=3.10 -y
conda activate climate-stage2
pip install -r requirements.txt
```

1) Download models (examples):
```
hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir /home/agi592/models/mistral-7b --local-dir-use-symlinks False
hf download sentence-transformers/all-mpnet-base-v2 --local-dir /home/agi592/models/all-mpnet-base-v2 --local-dir-use-symlinks False
```
2) Stage1 encoder/classifier (WeatherQA):
```
encoder:   /home/agi592/kse/ClimateToText/stage1_curriculum_runs/step1_all_types/stage1_vision_encoder_mae.pt
classifier:/home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt
```
3) Install deps (inside your venv):
```
pip install -r requirements.txt   # if not already done above
```
(Use this repo’s requirements; CUDA/torch must match your GPU.)

---

## Tasks (MTL)
- **A: Element Caption** — Image → 4‑sentence caption  
- **B: Image → Global Summary** — Image + cond_name → MD summary  
- **C: CoT Link** — Caption S1–S3 + global summary → S4 (link/explanation)  
- **D (optional): Captions → Summary** — List of captions → MD summary

---

## Training Recipes

> Set `CUDA_VISIBLE_DEVICES` as needed. Replace paths if different.  
> All commands assume `twkim/climate_to_text_stage2` as the working directory.

### Case 1: Uniform MTL (Tasks A/B/C mixed)
```
torchrun --nproc_per_node=2 --master_port 29510 \
  -m src.tasks.train_mtl_weatherqa \
  --gemini-json kse/ClimateToText/data/WeatherQA/gemini_element_captions_2014-2019.json \
                kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json \
  --image-root kse/ClimateToText/data/WeatherQA \
  --model-path /home/agi592/models/mistral-7b \
  --output-dir twkim/checkpoints/stage2_mtl_weatherqa_uniform \
  --batch-size 1 --num-epochs 1 \
  --lr 2e-5 --encoder-lr 5e-6 \
  --warmup-ratio 0.1 --max-seq-len 256 --image-size 224 \
  --use-lora --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --num-image-tokens 4 \
  --stage1-encoder-ckpt /home/agi592/kse/ClimateToText/stage1_curriculum_runs/step1_all_types/stage1_vision_encoder_mae.pt \
  --stage1-classifier-ckpt /home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt \
  --stage1-trainable-blocks 1 \
  --distributed --num-workers 2 \
  --sample-log-steps 500 --sample-log-max-new-tokens 80 \
  --use-wandb --wandb-project climate-to-text-stage2-mtl \
  --wandb-run-name "stage2_uniform_mtl"
```

### Case 2: Summary‑Centric MTL (warm‑up A+C → summary‑heavy)
```
torchrun --nproc_per_node=2 --master_port 29520 \
  -m src.tasks.train_summary_centric \
  --gemini-json kse/ClimateToText/data/WeatherQA/gemini_element_captions_2014-2019.json \
                kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json \
  --image-root kse/ClimateToText/data/WeatherQA \
  --model-path /home/agi592/models/mistral-7b \
  --output-dir twkim/checkpoints/stage2_summary_centric \
  --batch-size 1 \
  --warmup-epochs 1 --main-epochs 1 \
  --prob-summary 0.7 --prob-caption 0.15 --prob-cot 0.15 --prob-text-summary 0.0 \
  --lr 2e-5 --encoder-lr 5e-6 \
  --warmup-ratio 0.1 --max-seq-len 256 --image-size 224 \
  --use-lora --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --num-image-tokens 4 \
  --stage1-encoder-ckpt /home/agi592/kse/ClimateToText/stage1_curriculum_runs/step1_all_types/stage1_vision_encoder_mae.pt \
  --stage1-classifier-ckpt /home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt \
  --stage1-trainable-blocks 1 \
  --distributed --num-workers 2 \
  --sample-log-steps 500 --sample-log-max-new-tokens 80 \
  --use-wandb --wandb-project climate-to-text-stage2-mtl \
  --wandb-run-name "stage2_summary_centric"
```

*(Phased or D‑inclusive variants: use `train_summary_centric_phased.py` or enable `--text-summary` in the scripts.)*

---

## Inference

### Stage2 (LoRA + Stage1 encoder)
```
python -m src.inference.inference_simple \
  --json-path kse/ClimateToText/data/WeatherQA/gemini_element_captions_2014-2019.json \
  --image-root kse/ClimateToText/data/WeatherQA \
  --num-samples 3 \
  --max-new-tokens 128 --num-beams 1 \
  --no-repeat-ngram-size 4 --repetition-penalty 1.1 \
  --checkpoint-dir twkim/checkpoints/stage2_mtl_weatherqa_uniform \
  --base-model-path /home/agi592/models/mistral-7b \
  --stage1-encoder-ckpt /home/agi592/kse/ClimateToText/stage1_curriculum_runs/step1_all_types/stage1_vision_encoder_mae.pt \
  --stage1-classifier-ckpt /home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt
```

### Pretrained‑only baseline
```
python -m src.inference.inference_simple_pretrained \
  --json-path kse/ClimateToText/data/WeatherQA/gemini_element_captions_2014-2019.json \
  --image-root kse/ClimateToText/data/WeatherQA \
  --num-samples 3 \
  --max-new-tokens 128 --num-beams 1 \
  --no-repeat-ngram-size 4 --repetition-penalty 1.1 \
  --base-model-path /home/agi592/models/mistral-7b
```

### Compare pretrained vs baseline vs MTL
```
python -m src.inference.compare_inference_mtl \
  --json-path kse/ClimateToText/data/WeatherQA/gemini_element_captions_2014-2019.json \
  --image-root kse/ClimateToText/data/WeatherQA \
  --num-samples 3 \
  --max-new-tokens 128 --num-beams 1 \
  --no-repeat-ngram-size 4 --repetition-penalty 1.1 \
  --pretrained-model-path /home/agi592/models/mistral-7b \
  --baseline-ckpt twkim/checkpoints/stage2_mtl_weatherqa_uniform \
  --mtl-ckpt twkim/checkpoints/stage2_summary_centric \
  --stage1-encoder-ckpt /home/agi592/kse/ClimateToText/stage1_curriculum_runs/step1_all_types/stage1_vision_encoder_mae.pt \
  --stage1-classifier-ckpt /home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt
```

---

## Checkpoint Contents
- `adapter_model.bin` (LoRA)  
- `stage2_prefix_modules.pt` (class embeddings, image projection)  
- `tokenizer` files  
- `cond_name_to_id.json`  
- (optional) `stage1_encoder_finetuned.pt` (if Stage1 blocks were trainable)

---

## Notes
- Use AMP (default) to avoid OOM; FP32 often exceeds 24 GB.  
- `max_seq_len=256` is more stable; longer sequences (e.g., 512) may need smaller LR or more VRAM.  
- If you change Stage1 encoders/classifiers, keep `cond_name` mapping consistent.  
- wandb logging is optional; disable with `--use-wandb` removed.

This repo, with the commands above, is self‑contained for Stage2 MTL training, inference, and evaluation on the Gemini‑augmented WeatherQA dataset.***
