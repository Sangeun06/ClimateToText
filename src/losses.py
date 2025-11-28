from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F



def contrastive_loss(
    climate_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if climate_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D tensors.")
    logits = climate_embeddings @ text_embeddings.t()
    logits = logits / max(temperature, 1e-6)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return (loss_i + loss_t) * 0.5


def _pairwise_gaussian_kl(
    mean_a: torch.Tensor,
    logvar_a: torch.Tensor,
    mean_b: torch.Tensor,
    logvar_b: torch.Tensor,
) -> torch.Tensor:
    var_a = torch.exp(logvar_a).unsqueeze(1)
    var_b = torch.exp(logvar_b).unsqueeze(0)
    diff = mean_b.unsqueeze(0) - mean_a.unsqueeze(1)
    term_var = (var_a / var_b).sum(dim=-1)
    term_diff = (diff.pow(2) / var_b).sum(dim=-1)
    term_log = (logvar_b.unsqueeze(0) - logvar_a.unsqueeze(1)).sum(dim=-1)
    dim = mean_a.size(-1)
    return 0.5 * (term_var + term_diff + term_log - dim)


def masked_language_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    )
    if attention_mask is not None:
        loss = loss.view(labels.size(0), labels.size(1)) * attention_mask.float()
        denominator = attention_mask.sum()
    else:
        keep_mask = (labels.view(-1) != ignore_index).float()
        denominator = keep_mask.sum().clamp(min=1.0)
    return loss.sum() / denominator


def compute_mae_recon_loss(predictions_rgb, targets_rgb):
    """
    PerceiverQueryDecoder의 출력과 타겟 픽셀 간의 L1 Loss 계산
    """
    # L1 Loss를 사용하여 선명도 개선 시도 (L2(MSE) 대신)
    return F.l1_loss(predictions_rgb, targets_rgb)


@dataclass
class LossOutputs:
    total_loss: torch.Tensor
    contrastive: torch.Tensor
    decoder: Optional[torch.Tensor]