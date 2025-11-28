import math
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


JUDGE_PROMPT_TEMPLATE = """You are an expert meteorologist and writing quality judge.
Given two weather-related text descriptions A and B, output a similarity score between 0 and 1
based on their semantic content (0 = completely different, 1 = essentially identical).

Only output in the following format:
SCORE: <number between 0 and 1 with up to 3 decimal places>

A:
{a}

B:
{b}

SCORE:"""


class LLMSimilarityJudge:
    """
    Use a (local) LLM as a semantic similarity judge.

    Usage:
        judge = LLMSimilarityJudge("path-or-hf-id")
        score = judge.score("text a", "text b")  # in [0, 1]
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ):
        """
        Args:
            model_name_or_path: HF model ID or local path for a causal LLM
                               (e.g., Mistral-Instruct checkpoint directory).
            device: "cuda", "cpu", or None for auto-detect.
            max_new_tokens: Max tokens to generate for the score.
            temperature: Sampling temperature (0.0 = deterministic).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _build_prompt(self, a: str, b: str) -> str:
        return JUDGE_PROMPT_TEMPLATE.format(a=a, b=b)

    @staticmethod
    def _parse_score(text: str) -> float:
        """
        Parse "SCORE: x.xxx" from model output and return a float in [0, 1].
        """
        match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text)
        if not match:
            return 0.0
        value = float(match.group(1))
        # Clamp to [0, 1] just in case the model outputs >1.0
        value = max(0.0, min(1.0, value))
        if math.isnan(value):
            return 0.0
        return value

    @torch.no_grad()
    def score(self, a: str, b: str) -> float:
        """
        Ask the LLM to judge similarity between two texts.

        Returns:
            Similarity score in [0, 1] as judged by the LLM.
        """
        prompt = self._build_prompt(a, b)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0.0,
            temperature=self.temperature if self.temperature > 0.0 else 1.0,
        )

        # Decode full text (prompt + completion), then parse "SCORE: x.xxx".
        # The generated part typically only contains the numeric value,
        # while "SCORE:" is already in the prompt.
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self._parse_score(text)


def score_with_llm_judge(
    a: str,
    b: str,
    model_name_or_path: str,
    device: Optional[str] = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> float:
    """
    Convenience function to compute LLM-based similarity in one call.

    This will instantiate the LLM each time; for repeated calls,
    prefer using LLMSimilarityJudge directly.
    """
    judge = LLMSimilarityJudge(
        model_name_or_path=model_name_or_path,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return judge.score(a, b)
