from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


## 이미지 인코더 정의: PerceiverEncoder
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

class MAEDecoder(nn.Module):
    """
    단순 MLP 기반의 MAE 디코더
    (이전 대화의 "Simple MLP Decoder")
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

## 단순 분류기 (Stage 1용) - 이미지 종류 구분용
class Stage1Classifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)