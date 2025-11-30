"""
Stage1 (ClimateToText) 이미지 인코더 + 분류기를
Stage2 파이프라인에서 사용할 수 있도록 감싸는 헬퍼.

DeepWeatherInsight용으로 복사된 버전 (twkim 모듈 의존성 제거)
"""

from pathlib import Path
from typing import Optional, Tuple, Dict

import sys

import torch
from torch import nn

# Stage1 레포 루트 경로 (현재 환경 기준)
STAGE1_ROOT = Path("/home/agi592/twkim/ClimateToText")

if STAGE1_ROOT.exists():
    if str(STAGE1_ROOT) not in sys.path:
        sys.path.append(str(STAGE1_ROOT))
else:
    pass

try:
    from script.stage1_inference import Stage1InferencePipeline  # type: ignore
except Exception:  # pragma: no cover - Stage1 코드가 없는 환경
    Stage1InferencePipeline = None  # type: ignore


class Stage1ImageEncoderForStage2(nn.Module):
    """
    ClimateToText Stage1InferencePipeline을 Stage2에서 사용하는
    `image_encoder` 형태로 감싼 래퍼.

    Stage2 입장에서 인터페이스:
        - forward(images, cond_ids=None) -> z_img [B, latent_dim]
        - self.out_dim = latent_dim
        - self.num_conditions = Stage1에서 사용하는 이미지 타입 개수
    """

    def __init__(
        self,
        encoder_ckpt_path: str,
        classifier_ckpt_path: str,
        device: Optional[torch.device] = None,
        freeze: bool = True,
        num_trainable_blocks: int = 0,
    ) -> None:
        super().__init__()

        if Stage1InferencePipeline is None:
            raise ImportError(
                "Stage1InferencePipeline을 import하지 못했습니다. "
                "ClimateToText 레포가 /home/agi592/twkim/ClimateToText 에 있고, "
                "필요한 의존성이 설치되어 있는지 확인하세요."
            )

        encoder_ckpt_path = str(encoder_ckpt_path)
        classifier_ckpt_path = str(classifier_ckpt_path)

        # Stage1 encoder 설정은 Stage1 학습 시 사용한 값과 동일해야 한다.
        self.latent_dim = 768
        self.num_conditions = 20
        self.pipeline = Stage1InferencePipeline(
            patch_size=16,
            latent_dim=self.latent_dim,
            num_latents=196,
            num_blocks=6,
            num_heads=8,
            moe_num_experts=8,
            num_conditions=self.num_conditions,
            classifier_backbone="efficientnet_b0",
            encoder_ckpt_path=encoder_ckpt_path,
            classifier_ckpt_path=classifier_ckpt_path,
            encoder_return_all=False,
            encoder_mode="clip",
        )

        # classifier는 확실히 freeze
        if hasattr(self.pipeline, "classifier"):
            for p in self.pipeline.classifier.parameters():
                p.requires_grad = False

        if freeze or num_trainable_blocks <= 0:
            # 완전 freeze 모드
            self.pipeline.eval()
            for p in self.pipeline.parameters():
                p.requires_grad = False
        else:
            # encoder 전체를 먼저 freeze한 뒤, 마지막 num_trainable_blocks만 unfreeze
            enc = getattr(self.pipeline, "encoder", None)
            if enc is not None:
                for p in enc.parameters():
                    p.requires_grad = False

                blocks = getattr(enc, "blocks", None)
                if blocks is not None and len(blocks) > 0:
                    k = min(num_trainable_blocks, len(blocks))
                    for block in blocks[-k:]:
                        for p in block.parameters():
                            p.requires_grad = True

                # latent 토큰 자체도 약간 조정 가능하게 하고 싶다면:
                if hasattr(enc, "latents"):
                    enc.latents.requires_grad = True

        self.out_dim = self.latent_dim

        if device is not None:
            self.to(device)

    def encode_and_classify(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage1 파이프라인을 실행해 latent와 Stage1 cond_ids를 동시에 반환.
        """
        latents, pred_cond_ids = self.pipeline(images)
        return latents, pred_cond_ids

    def forward(self, images: torch.Tensor, cond_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images  : [B, 3, H, W] 이미지 텐서.
            cond_ids: Stage2 class id (이 래퍼에서는 사용하지 않는다).

        Returns:
            z_img   : [B, latent_dim] Stage1 encoder latent.
        """
        latents, _ = self.pipeline(images)
        return latents


class ImageToTextModelStage1(nn.Module):
    """
    Stage1 이미지 인코더 + 분류기를 사용하는 Stage2 모델.

    - 이미지 → Stage1 pipeline → (z_img, cond_ids_stage1)
    - cond_ids_stage1를 class token 생성에 사용
    - 텍스트 부분은 기본 ImageToTextModel과 동일한 구조
    """

    def __init__(
        self,
        llm: nn.Module,
        image_encoder: Stage1ImageEncoderForStage2,
        num_image_tokens: int = 4,
    ) -> None:
        super().__init__()

        self.llm = llm
        self.image_encoder = image_encoder
        self.num_image_tokens = num_image_tokens

        hidden_size = llm.config.hidden_size
        llm_dtype = next(llm.parameters()).dtype

        # Stage1 cond_id 개수만큼 class embedding
        num_classes = image_encoder.num_conditions
        self.class_embeddings = nn.Embedding(num_classes, hidden_size)
        self.class_embeddings.to(llm_dtype)

        img_latent_dim = image_encoder.latent_dim
        self.image_projection = nn.Sequential(
            nn.Linear(img_latent_dim, hidden_size * num_image_tokens),
            nn.GELU(),
            nn.Linear(hidden_size * num_image_tokens, hidden_size * num_image_tokens),
        )
        self.image_projection.to(llm_dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        pixel_values: torch.Tensor,
        cond_ids: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        cond_ids 인자는 무시하고, Stage1 classifier에서 나온 ID를 사용한다.
        """
        device = self.device
        llm_dtype = next(self.llm.parameters()).dtype

        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        batch_size = pixel_values.size(0)

        # Stage1 pipeline: 이미지 → latent + cond_ids_stage1
        z_img, cond_ids_stage1 = self.image_encoder.encode_and_classify(pixel_values)
        z_img = z_img.to(llm_dtype)
        cond_ids_stage1 = cond_ids_stage1.to(device)

        hidden_size = self.llm.config.hidden_size
        img_prefix = self.image_projection(z_img)
        img_prefix = img_prefix.view(batch_size, -1, hidden_size)
        img_prefix = img_prefix.to(llm_dtype)

        class_prefix = self.class_embeddings(cond_ids_stage1)
        class_prefix = class_prefix.unsqueeze(1)
        class_prefix = class_prefix.to(llm_dtype)

        prefix_embeddings = torch.cat([img_prefix, class_prefix], dim=1)
        num_prefix_tokens = prefix_embeddings.size(1)

        input_embeddings = self.llm.get_input_embeddings()(input_ids)
        input_embeddings = input_embeddings.to(llm_dtype)
        inputs_embeds = torch.cat([prefix_embeddings, input_embeddings], dim=1)

        prefix_attention = torch.ones(
            batch_size,
            num_prefix_tokens,
            dtype=attention_mask.dtype,
            device=device,
        )
        extended_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        if labels is None:
            labels = input_ids.masked_fill(attention_mask == 0, -100)

        prefix_labels = torch.full(
            (batch_size, num_prefix_tokens),
            -100,
            dtype=torch.long,
            device=device,
        )
        extended_labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        cond_ids: Optional[torch.Tensor],
        tokenizer,
        generation_prompt: str = "",
        max_new_tokens: int = 128,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
    ) -> Tuple[str, ...]:
        """
        cond_ids 인자는 무시하고 Stage1 classifier 결과를 사용해 class token을 만든다.
        """
        self.eval()
        device = self.device

        pixel_values = pixel_values.to(device)
        llm_dtype = next(self.llm.parameters()).dtype

        batch_size = pixel_values.size(0)

        z_img, cond_ids_stage1 = self.image_encoder.encode_and_classify(pixel_values)
        z_img = z_img.to(llm_dtype)
        cond_ids_stage1 = cond_ids_stage1.to(device)

        hidden_size = self.llm.config.hidden_size
        img_prefix = self.image_projection(z_img)
        img_prefix = img_prefix.view(batch_size, -1, hidden_size)
        img_prefix = img_prefix.to(llm_dtype)

        class_prefix = self.class_embeddings(cond_ids_stage1)
        class_prefix = class_prefix.unsqueeze(1)
        class_prefix = class_prefix.to(llm_dtype)

        prefix_embeddings = torch.cat([img_prefix, class_prefix], dim=1)
        num_prefix_tokens = prefix_embeddings.size(1)

        if generation_prompt.strip() == "":
            prompt_texts = [""] * batch_size
        else:
            prompt_texts = [generation_prompt] * batch_size

        tokenized = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        text_embeddings = text_embeddings.to(llm_dtype)

        inputs_embeds = torch.cat([prefix_embeddings, text_embeddings], dim=1)

        prefix_attention = torch.ones(
            batch_size,
            num_prefix_tokens,
            dtype=attention_mask.dtype,
            device=device,
        )
        extended_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        generate_kwargs = dict(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,  # 프롬프트 토큰을 출력 시퀀스에 포함시켜 앞단 잘림 방지
            attention_mask=extended_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if repetition_penalty and repetition_penalty != 1.0:
            generate_kwargs["repetition_penalty"] = repetition_penalty

        generated = self.llm.generate(**generate_kwargs)

        prompt_len = input_ids.shape[1]
        generated_texts = []
        for seq in generated:
            text_tokens = seq[prompt_len:]
            text = tokenizer.decode(
                text_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            text = text.strip()
            last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_punct != -1:
                text = text[: last_punct + 1].strip()
            generated_texts.append(text)

        return tuple(generated_texts)
