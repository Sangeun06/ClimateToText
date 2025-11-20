import math
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class EmbeddingSimilarityScorer:
    """
    Simple text similarity scorer based on a pretrained text encoder.

    Usage:
        scorer = EmbeddingSimilarityScorer("path-or-hf-id")
        score = scorer.score("text a", "text b")  # cosine similarity in [-1, 1]
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        normalize: bool = True,
        max_length: int = 512,
    ):
        """
        Args:
            model_name_or_path: HF model ID or local path for a text encoder
                               (e.g., "sentence-transformers/all-MiniLM-L6-v2"
                               or a local directory).
            device: "cuda", "cpu", or None for auto-detect.
            normalize: If True, return cosine similarity in [-1, 1].
            max_length: Max token length for the encoder.
        """
        # Decide whether to use SentenceTransformer (if available) or plain AutoModel.
        self.normalize = normalize
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Accept both strings and torch.device
        self.device = torch.device(device)

        self._use_sentence_transformer = False
        self._st_model = None

        # Prefer SentenceTransformer if it's installed and can load the checkpoint.
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(
                model_name_or_path, device=str(self.device)
            )
            self._use_sentence_transformer = True
            self.tokenizer = None
            self.model = None
        except (ImportError, OSError):
            # Fallback: plain HF encoder
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(model_name_or_path)
            self.model.to(self.device)
            self.model.eval()

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts into embeddings.

        Returns:
            Tensor of shape [B, D].
        """
        if self._use_sentence_transformer and self._st_model is not None:
            # SentenceTransformer handles pooling internally.
            # normalize_embeddings controls L2-normalization.
            emb = self._st_model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize,
            )
            return emb.to(self.device)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self.model(**encoded)

        # CLS pooling if available, otherwise mean pooling over tokens
        if hasattr(outputs, "last_hidden_state"):
            token_embeddings = outputs.last_hidden_state  # [B, T, D]
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            embeddings = summed / counts
        else:
            # Fallback: try "pooler_output" or just use outputs[0]
            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs[0]

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @torch.no_grad()
    def score(self, a: str, b: str) -> float:
        """
        Compute similarity between two text descriptions.

        Returns:
            Cosine similarity (float). If normalize=True, in [-1, 1].
        """
        embeddings = self._encode([a, b])  # [2, D]
        emb_a, emb_b = embeddings[0], embeddings[1]
        sim = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0), dim=-1)
        value = sim.item()

        # Guard against tiny numeric issues
        if math.isnan(value):
            return 0.0
        return float(value)


def score_with_embedding_model(
    a: str,
    b: str,
    model_name_or_path: str,
    device: Optional[str] = None,
) -> float:
    """
    Convenience function to compute similarity in one call.

    This will instantiate the encoder each time; for repeated calls,
    prefer using EmbeddingSimilarityScorer directly.
    """
    scorer = EmbeddingSimilarityScorer(model_name_or_path=model_name_or_path, device=device)
    return scorer.score(a, b)
