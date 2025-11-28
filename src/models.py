from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True, output_dim=768):
        super().__init__()

        # 1. 사전학습된 ResNet-50 로드
        # (weights=models.ResNet50_Weights.IMAGENET1K_V1 과 동일)
        resnet = models.resnet50(pretrained=pretrained)

        # 2. ResNet의 마지막 FC 레이어와 Pooling 레이어 제거
        #    (공간 정보를 유지하기 위해 layer4까지만 사용)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # 3. ResNet-50의 출력 채널(2048)을 우리가 원하는 latent_dim(예: 768)으로 맞춤
        self.proj = nn.Linear(2048, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        # x: (B, 3, 224, 224)

        # 1. Backbone 통과 -> 특징 맵 추출
        # 출력: (B, 2048, 7, 7) (224x224 입력 시)
        features = self.backbone(x)

        # 2. Flatten & Permute (Transformer/Decoder 입력 형태로 변환)
        # (B, 2048, 7, 7) -> (B, 2048, 49) -> (B, 49, 2048)
        features = features.flatten(2).transpose(1, 2)

        # 3. 차원 축소 (2048 -> 768)
        # (B, 49, 768)
        latents = self.proj(features)

        return latents # (B, L=49, D=768)


## Image 전처리: 이미지를 패치로 나누고, 1차원 시퀀스로 펼침
class ImagePreprocessor(nn.Module):
    """
    Processes a batch of images into patches and flattens them for Perceiver-style models.
    """

    def __init__(self, patch_size: int = 16, latent_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.projection = nn.Conv2d(3, latent_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Batch of images with shape (B, C, H, W).

        Returns:
            torch.Tensor: Flattened patch embeddings with shape (B, N, D),
                          where N is the number of patches and D is the latent dimension.
        """
        patches = self.projection(images)  # (B, D, H', W')
        patches = patches.flatten(2)  # (B, D, N)
        patches = patches.transpose(1, 2)  # (B, N, D)
        return patches


## 트랜스포머 블록 정의: Cross-Attention, MoE, Perceiver Block
class CrossAttentionLayer(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(latent_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # Query: (B, L_q, D), Key/Value: (B, L_kv, D)
        attn_output, _ = self.attention(query, key_value, key_value)
        query = self.norm1(query + attn_output)
        return query

class MoELayer(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim)) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        routing_weights = F.softmax(self.router(x), dim=-1)  # (B, N, num_experts)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # (B, N, k)

        # Normalize the weights for the top-k experts
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        final_output = torch.zeros_like(x)

        for i in range(self.top_k):
            expert_indices = top_k_indices[..., i]  # (B, N)
            expert_weights = top_k_weights[..., i].unsqueeze(-1)  # (B, N, 1)

            for j in range(len(self.experts)):
                mask = (expert_indices == j)  # (B, N)
                if mask.any():
                    tokens_for_expert = x[mask]  # (num_tokens, D)
                    expert_output = self.experts[j](tokens_for_expert)
                    final_output[mask] += expert_output * expert_weights[mask]

        return final_output

class PerceiverBlock(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int, moe_num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.moe_layer = MoELayer(hidden_dim=latent_dim, num_experts=moe_num_experts)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)
        moe_output = self.moe_layer(x)
        x = self.norm2(x + moe_output)
        return x


## 이미지 인코더 정의: PerceiverEncoder, ConditionalPerceiverEncoder
class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        latent_dim: int = 768,
        num_latents: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        moe_num_experts: int = 8,
        dropout: float = 0.1,
        num_conditions: int = 0,
    ):
        super().__init__()
        self.preprocessor = ImagePreprocessor(patch_size=patch_size, latent_dim=latent_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attention = CrossAttentionLayer(latent_dim, num_heads, dropout)
        self.blocks = nn.ModuleList([PerceiverBlock(latent_dim, num_heads, moe_num_experts, dropout) for _ in range(num_blocks)])
        self.output_dim = latent_dim
        # Optional conditioning (dataset/type-aware)
        self.num_conditions = int(num_conditions) if num_conditions is not None else 0
        if self.num_conditions and self.num_conditions > 0:
            self.cond_embedding = nn.Embedding(self.num_conditions, latent_dim)
        else:
            self.cond_embedding = None

    def forward(self, x: torch.Tensor, return_all: bool = False, cond_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, H, W)
        patches = self.preprocessor(x)  # (B, N, D)
        batch_size = patches.size(0)

        # Broadcast latents to batch size
        latents = self.latents.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, L, D)

        # Apply simple additive conditioning on latents, if provided
        if self.cond_embedding is not None and cond_ids is not None:
            # cond_ids: (B,) -> (B, D)
            cond_vec = self.cond_embedding(cond_ids).unsqueeze(1)  # (B, 1, D)
            latents = latents + cond_vec  # broadcast add

        # Cross-attention from latents to patches
        latents = self.cross_attention(latents, patches)

        # Self-attention blocks
        for block in self.blocks:
            latents = block(latents)
        return latents if return_all else latents.mean(dim=1)

class ConditionalPerceiverEncoder(nn.Module):
    """
    '이미지 타입'을 조건(condition)으로 받는 MAE용 Perceiver 인코더.
    오직 '보이는' 패치만 입력받습니다.
    """
    def __init__(
        self,
        num_types: int, # ★ 추가: 총 이미지 타입 개수
        patch_size: int = 16,
        latent_dim: int = 768,
        num_latents: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        moe_num_experts: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1. 이미지 패치 처리기
        self.preprocessor = ImagePreprocessor(patch_size=patch_size, latent_dim=latent_dim)

        # 2. Perceiver 핵심
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attention = CrossAttentionLayer(latent_dim, num_heads, dropout)
        self.blocks = nn.ModuleList([PerceiverBlock(latent_dim, num_heads, moe_num_experts, dropout) for _ in range(num_blocks)])

        # 3. ★ 조건부(Conditional)를 위한 타입 임베딩
        self.type_embedding = nn.Embedding(num_types, latent_dim)

        # 4. MAE를 위한 위치 임베딩 (학습 가능)
        #    (이미지 크기가 224x224, 패치 16x16 -> 14x14=196개 패치라고 가정)
        num_patches = (224 // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, latent_dim))

        self.output_dim = latent_dim

    def forward(self, images: torch.Tensor, type_ids: torch.Tensor, keep_ratio: float = 0.25):
        """
        MAE의 forward pass.
        Args:
            images (torch.Tensor): (B, C, H, W) - 원본 이미지 (아직 마스킹되지 않음)
            type_ids (torch.Tensor): (B,) - 각 이미지의 타입 ID
            keep_ratio (float): 마스킹하지 않고 '볼' 패치의 비율 (예: 25%)
        """
        b = images.shape[0]

        # 1. 패치화 및 위치 임베딩
        patches = self.preprocessor(images) # (B, N, D) (N=196)
        patches = patches + self.pos_embedding # (B, N, D)

        # 2. ★ 타입 임베딩 융합
        # (B,) -> (B, D) -> (B, 1, D)
        task_embed = self.type_embedding(type_ids).unsqueeze(1)
        # (B, N, D) + (B, 1, D) -> (B, N, D)
        patches = patches + task_embed # 모든 패치에 '타입 정보'를 더해줌

        # 3. 마스킹 (MAE 핵심)
        n_visible = int(patches.shape[1] * keep_ratio)
        visible_patches, visible_indices, masked_indices = self.random_masking(patches, n_visible)

        # 4. Perceiver 인코딩 (오직 '보이는' 패치만 입력)
        # (B, L, D) - L=num_latents
        latents = self.latents.unsqueeze(0).repeat(b, 1, 1)
        latents = self.cross_attention(latents, visible_patches)
        for block in self.blocks:
            latents = block(latents)

        # (latents.mean(dim=1)을 하지 않고 전체 latents 반환)
        return latents, visible_indices, masked_indices

    def random_masking(self, x, n_visible):
        """[Helper] 패치를 무작위로 섞고, 보이는/가려진 패치로 나눔"""
        b, n, d = x.shape
        # 무작위 인덱스 생성
        rand_indices = torch.rand(b, n, device=x.device).argsort(dim=1)

        visible_indices = rand_indices[:, :n_visible]
        masked_indices = rand_indices[:, n_visible:]

        # 인덱스를 사용하여 패치 선택
        visible_patches = torch.gather(x, dim=1, index=visible_indices.unsqueeze(-1).expand(-1, -1, d))

        return visible_patches, visible_indices, masked_indices

# [Helper] 2D Sine-Cosine Positional Embedding 생성 함수
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size_h, grid_size_w에 맞춰서 즉석에서 포지션 임베딩을 계산합니다.
    학습 파라미터가 아니므로 해상도가 바뀌어도 상관없습니다.
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# --------------------------------------------------------------------------

class DynamicPerceiverEncoder(nn.Module):
    def __init__(
        self,
        # num_types: 제거하거나, 아주 큰 값으로 설정
        patch_size: int = 16,
        latent_dim: int = 768,
        num_latents: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        moe_num_experts: int = 8,
        dropout: float = 0.1,
        max_num_types: int = 1000, # [수정] 넉넉하게 잡아둠 (메모리 거의 안 먹음)
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim

        # 1. 이미지 패치 처리기 (Conv2d 등)
        # padding='valid'로 설정하면 이미지가 딱 떨어지지 않아도 처리 가능
        self.patch_embed = nn.Conv2d(3, latent_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Perceiver 핵심 (Latent는 고정 크기 유지)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attention = CrossAttentionLayer(latent_dim, num_heads, dropout)
        self.blocks = nn.ModuleList([
            PerceiverBlock(latent_dim, num_heads, moe_num_experts, dropout)
            for _ in range(num_blocks)
        ])

        # 3. [수정] 조건 임베딩 (넉넉한 버퍼)
        # 만약 정말로 무한한 타입을 원하면, type_ids 대신 'type_vector'를 입력받는 Linear로 교체해야 함
        self.type_embedding = nn.Embedding(max_num_types, latent_dim)

        # 4. [수정] 고정 위치 임베딩 삭제됨
        # self.pos_embedding = ... (삭제)

        self.output_dim = latent_dim

    def forward(self, images: torch.Tensor, type_ids: torch.Tensor, keep_ratio: float = 0.25):
        """
        images: (B, C, H, W) - H, W가 매번 달라도 됨
        """
        b, c, h, w = images.shape
        p = self.patch_size

        # 1. 패치화 (Conv2d 이용)
        # (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)

        # 2. [핵심 수정] 2D Sin-Cos Positional Embedding 동적 생성
        grid_h, grid_w = h // p, w // p

        # 현재 배치의 해상도에 맞는 임베딩 계산 (numpy -> tensor)
        pos_embed = get_2d_sincos_pos_embed(self.latent_dim, grid_h, grid_w, cls_token=False)
        pos_embed = torch.from_numpy(pos_embed).float().to(images.device)
        # (N, D) -> (1, N, D) Broadcasting
        pos_embed = pos_embed.unsqueeze(0)

        # 위치 정보 더하기
        patches = patches + pos_embed

        # 3. 타입 임베딩 융합
        task_embed = self.type_embedding(type_ids).unsqueeze(1) # (B, 1, D)
        patches = patches + task_embed

        # 4. 마스킹
        n_patches = patches.shape[1]
        n_visible = int(n_patches * keep_ratio)

        # (중요) n_visible이 0이 되지 않도록 최소 1개 보장
        n_visible = max(1, n_visible)

        visible_patches, visible_indices, masked_indices = self.random_masking(patches, n_visible)

        # 5. Perceiver 인코딩
        latents = self.latents.unsqueeze(0).repeat(b, 1, 1)
        latents = self.cross_attention(latents, visible_patches)
        for block in self.blocks:
            latents = block(latents)

        return latents, visible_indices, masked_indices

    def random_masking(self, x, n_visible):
        b, n, d = x.shape
        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)

        ids_restore = torch.argsort(ids_shuffle, dim=1) # 복원용 인덱스

        ids_keep = ids_shuffle[:, :n_visible]
        ids_mask = ids_shuffle[:, n_visible:]

        # gather로 선택
        visible_patches = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))

        return visible_patches, ids_keep, ids_mask

class MAEDecoder(nn.Module):
    """
    단순 MLP 기반의 MAE 디코더
    """
    def __init__(self, encoder_dim=768, num_latents=256, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        decoder_dim = 512 # 디코더는 인코더보다 가벼워도 됨

        # Perceiver 출력(latents)을 디코더 차원으로 프로젝션
        self.input_proj = nn.Linear(encoder_dim * num_latents, decoder_dim)

        # 디코더 MLP
        self.decoder = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 2),
            nn.GELU(),
            nn.Linear(decoder_dim * 2, decoder_dim * 4),
            nn.GELU(),
        )

        # 최종 픽셀로 복원
        num_patches = (224 // patch_size) ** 2
        pixel_dim = (patch_size ** 2) * 3
        self.output_proj = nn.Linear(decoder_dim * 4, num_patches * pixel_dim)

    def forward(self, latents: torch.Tensor):
        # latents: (B, L, D_enc) - (B, 256, 768)

        # 1. ★과도한 병목★ (이전 대화에서 지적된 부분)
        #    (B, L, D_enc) -> (B, L * D_enc)
        x = latents.flatten(1)

        # 2. MLP 통과
        x = self.input_proj(x)
        x = self.decoder(x)

        # 3. 픽셀로 복원
        x = self.output_proj(x) # (B, N * P*P*C)

        return x

def unpatchify(x, patch_size=16):
    """[Helper] (B, N, P*P*C) 텐서를 (B, C, H, W) 이미지로 복원"""
    b, n, ppc = x.shape
    c = 3
    p = patch_size
    h_w = int(n**0.5) # N이 제곱수(196)라고 가정 (H=W=14)

    x = x.reshape(b, h_w, h_w, p, p, c)
    # (B, H, W, P, P, C) -> (B, C, H, P, W, P)
    x = x.permute(0, 5, 1, 3, 2, 4)
    # (B, C, H*P, W*P)
    images = x.reshape(b, c, h_w * p, h_w * p)
    return images

class SimpleDecoder(nn.Module):
    """
    PerceiverEncoder의 글로벌 latent(벡터)를 받아 원본 이미지를 복원하도록 학습하는 간단 디코더.
    (진짜 MAE처럼 패치 단위 마스킹과 복원을 완전히 재현하지는 않지만, 입력 일부를 영(0)으로 마스킹하여
     복원하도록 유도하는 형태로 구현)
    """

    def __init__(self, latent_dim: int = 768, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        out_dim = 3 * image_size * image_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, out_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b = latent.size(0)
        x = self.net(latent)
        x = x.view(b, 3, self.image_size, self.image_size)
        return x

class PatchDecoder(nn.Module):
    """
    Perceiver의 모든 latent 토큰(B, L, D)을 받아 각 토큰을 하나의 이미지 패치로 복원한 뒤
    (h x w) 그리드로 재배열하여 전체 이미지를 구성합니다.

    전제: L == (image_size / patch_size) ** 2 인 경우가 가장 자연스럽습니다.
    만약 L이 정확히 일치하지 않으면, h = w = int(sqrt(L))로 맞추고 최종 H=W=h*patch_size 크기로 복원한 뒤
    필요 시 target image_size로 보간합니다.
    """

    def __init__(self, latent_dim: int, num_latents: int, patch_size: int = 16, image_size: int = 224, out_channels: int = 3, hidden_dim: Optional[int] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.patch_size = patch_size
        self.image_size = image_size
        self.out_channels = out_channels
        hdim = hidden_dim or latent_dim
        self.to_patch = nn.Sequential(
            nn.Linear(latent_dim, hdim),
            nn.GELU(),
            nn.Linear(hdim, out_channels * patch_size * patch_size),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: (B, L, D)
        if latents.dim() != 3:
            raise ValueError(f"PatchDecoder expects (B, L, D), got {tuple(latents.shape)}")
        b, l, d = latents.shape
        # 각 토큰을 패치로 투영
        patches = self.to_patch(latents)  # (B, L, C*p*p)
        c = self.out_channels
        p = self.patch_size
        # (B, L, C, p, p)
        patches = patches.view(b, l, c, p, p)
        # 그리드 크기 계산
        grid = int(math.sqrt(l))
        if grid * grid != l:
            # L이 완전 제곱이 아니면 가능한 최대 정사각형으로 재배열하고 남는 토큰은 무시
            usable = grid * grid
            patches = patches[:, :usable]
        h = w = int(math.sqrt(patches.size(1)))
        # (B, h, w, C, p, p) -> (B, C, h*p, w*p)
        patches = patches.view(b, h, w, c, p, p)
        img = patches.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, h * p, w * p)
        # 필요시 target image_size로 보간
        if (h * p) != self.image_size or (w * p) != self.image_size:
            img = F.interpolate(img, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return img


class PatchCnnRefineDecoder(nn.Module):
    """PatchDecoder 출력 이미지를 작은 CNN으로 한 번 더 보정하는 디코더.

    1단계: Perceiver latents -> PatchDecoder 를 통해 대략적인 이미지 복원
    2단계: 작은 CNN (Residual-like 블록)을 통과시켜 곡선/엣지/기호 등을 더 부드럽게 복원
    """

    def __init__(
        self,
        latent_dim: int,
        num_latents: int,
        patch_size: int = 16,
        image_size: int = 224,
        out_channels: int = 3,
        hidden_channels: int = 64,
    ):
        super().__init__()
        # 1단계: 기존 PatchDecoder로 coarse reconstruction
        self.patch_decoder = PatchDecoder(
            latent_dim=latent_dim,
            num_latents=num_latents,
            patch_size=patch_size,
            image_size=image_size,
            out_channels=
            out_channels,
        )

        # 2단계: 얕은 CNN refinement 모듈 (Residual 스타일)
        self.refine_net = nn.Sequential(
            nn.Conv2d(out_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # 1단계: coarse reconstruction
        coarse = self.patch_decoder(latents)  # (B, C, H, W)
        # 2단계: CNN refinement + residual
        refine = self.refine_net(coarse)
        return coarse + refine


class FourierFeatureEncoding(nn.Module):
    """
    해상도에 의존하지 않는 NeRF 스타일의 Fourier Encoding
    """
    def __init__(self, num_bands: int, include_pi: bool = True):
        super().__init__()
        self.num_bands = num_bands

        # 1. 해상도 값을 없애고, 2의 거듭제곱(Log Scale)으로 주파수 생성
        # 예: [1, 2, 4, 8, 16, 32, ...]
        # 이렇게 하면 낮은 주파수(전체적인 형태)와 높은 주파수(세밀한 엣지)를 골고루 학습합니다.
        freqs = 2.0 ** torch.arange(num_bands)

        # 2. Pi를 곱해주면 좌표 범위 [-1, 1]이 주파수 주기와 딱 맞아떨어지게 됩니다.
        # [-1, 1] 범위에서 sin(pi * x)는 반 파장, sin(2*pi*x)는 한 파장을 형성합니다.
        if include_pi:
            freqs = freqs * np.pi

        self.register_buffer("freqs", freqs) # (num_bands,)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, N_pixels, 2) - 값 범위는 [-1, 1] 필수

        # (B, N, 2, 1) * (F,) -> (B, N, 2, F)
        x = coords.unsqueeze(-1) * self.freqs

        # sin, cos 연결: (B, N, 2, 2*F) -> (B, N, 4*F)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        return x.reshape(coords.shape[0], coords.shape[1], -1)


class PerceiverQueryDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_latents: int, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.out_channels = out_channels # 3 (RGB), 1 (Gray), 4 (RGBD) 등 설정 가능

        num_freq_bands = 64
        query_dim = 4 * num_freq_bands
        self.fourier_encoding = FourierFeatureEncoding(num_freq_bands)
        self.query_proj = nn.Linear(query_dim, latent_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            batch_first=True
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, out_channels)
        )

    def forward(self,
                latents: torch.Tensor,
                pixel_coords: Optional[torch.Tensor] = None,
                output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args:
            latents (B, L, D): 인코더 Latents
            pixel_coords (B, N, 2): 학습 시 사용할 특정 픽셀 좌표들 ([-1, 1] 범위)
            output_size (H, W): 추론 시 생성하고 싶은 이미지의 크기 (pixel_coords가 None일 때 필수)
        """
        B, L, D = latents.shape
        device = latents.device

        # 1. 쿼리 좌표(coords) 결정
        if pixel_coords is None:
            # [수정 2] 추론 시 output_size를 받아서 그리드 생성
            if output_size is None:
                raise ValueError("Inference Mode(pixel_coords=None)에서는 output_size=(H, W)가 필요합니다.")

            H, W = output_size

            # 어떤 해상도가 들어오든 -1 ~ 1 사이의 좌표로 정규화하여 생성
            ys = torch.linspace(-1, 1, H, device=device)
            xs = torch.linspace(-1, 1, W, device=device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

            # (B, H*W, 2)
            coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2).expand(B, -1, -1)
        else:
            # 학습 모드: 이미 샘플링된 좌표 사용
            coords = pixel_coords

        # 2. 좌표 인코딩 (해상도 무관하게 [-1, 1] 좌표만 있으면 됨)
        queries = self.fourier_encoding(coords)
        queries = self.query_proj(queries)

        # 3. Cross Attention (Latent -> Pixel)
        pixels_feat, _ = self.cross_attn(query=queries, key=latents, value=latents)

        # 4. 픽셀 값 예측
        rgb = self.output_proj(pixels_feat)

        # 5. 반환 형태
        if pixel_coords is None:
            # 요청한 H, W 크기로 Reshape하여 반환
            return rgb.view(B, H, W, self.out_channels).permute(0, 3, 1, 2)
        else:
            return rgb


## 단순 분류기 (Stage 1용) - 이미지 종류 구분용
class Stage1Classifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# =============================
# CLIP/ViT 기반 이미지 인코더
# =============================
from transformers import CLIPVisionModel, ViTModel


class CLIPViTEncoder(nn.Module):
    """
    CLIP/ViT 기반 이미지 인코더.
    - CLIP: 이미지와 텍스트를 동시에 임베딩
    - ViT: 순수 이미지 임베딩
    """
    def __init__(self, model_name="openai/clip-vit-base-patch16", output_dim=768, use_clip=True):
        super().__init__()
        self.use_clip = use_clip
        if use_clip:
            if CLIPVisionModel is None:
                raise ImportError("transformers 라이브러리의 CLIPModel이 필요합니다.")
            self.model = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True)
            self.output_dim = output_dim
        else:
            if ViTModel is None:
                raise ImportError("transformers 라이브러리의 ViTModel이 필요합니다.")
            self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", use_safetensors=True)
            self.output_dim = self.model.config.hidden_size
        # Optional: projection to match output_dim if needed
        if output_dim != self.output_dim:
            self.proj = nn.Linear(self.output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()

    def forward(self, x, patchwise=True):
        # x: (B, 3, H, W), pixel values normalized to [0, 1]
        if self.use_clip:
            if patchwise:
                # [수정 1] vision_model 호출
                outputs = self.model.vision_model(
                    pixel_values=x,
                    output_hidden_states=True,
                    #return_dict=True  # 명시적으로 Dict 반환 요청
                )

                # [수정 2] hidden_states[0](입력) 대신 last_hidden_state(출력) 사용
                # last_hidden_state shape: (B, Sequence_Length, Hidden_Size)
                last_hidden_state = outputs.last_hidden_state

                # [수정 3] CLS 토큰 제거 (첫 번째 토큰 제외)
                # CLIP의 경우 0번이 CLS 토큰, 1번부터가 이미지 패치입니다.
                patch_embeds = last_hidden_state[:, 1:, :]

                return self.proj(patch_embeds)

            else:
                # [주의] 이 경로는 Stage 2 학습 시 사용하면 안 됩니다! (토큰 1개만 나감)
                outputs = self.model.get_image_features(pixel_values=x)
                return self.proj(outputs)

        else:
            # ViT 로직도 동일하게 last_hidden_state 사용
            if patchwise:
                outputs = self.model(
                    pixel_values=x,
                    output_hidden_states=True,
                    return_dict=True
                )
                # ViT도 0번 인덱스가 CLS 토큰입니다.
                patch_embeds = outputs.last_hidden_state[:, 1:, :]
                return self.proj(patch_embeds)
            else:
                outputs = self.model(x).pooler_output
                return self.proj(outputs)