"""
Module 3 — GatingNetwork / StackingHead

두 가지 동적 앙상블 모듈:

GatingNetwork (model_type="gating"):
    z_list: [(B, z_dim)] × N  → stack → (B, N, z_dim)
    context = mean(stacked, dim=1)           # (B, z_dim)
    gate_weights = activate(gate(context))   # (B, N)
    z_fused = sum(weights * stacked, dim=1)  # (B, z_dim)
    logits = classifier(dropout(z_fused))    # (B, C)

StackingHead (model_type="stacking"):
    logits_list: [(B, C)] × N  → softmax → cat → (B, N*C)
    logits = MLP(probs_concat)               # (B, C)

    z 대신 각 모델의 softmax 확률을 직접 입력으로 사용.
    분류 결정에 더 직접적인 신호 → 적은 파라미터로 높은 성능.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GatingNetwork(nn.Module):
    """
    N개 모델의 z 임베딩을 컨텍스트 기반 가중합으로 융합하고 logits를 출력.

    Args:
        z_dim:       각 모델의 latent embedding 차원.
        num_sources: 융합할 모델 수 (N).
        hidden_size: gate MLP 내부 hidden 차원.
        num_classes: 출력 클래스 수 (C).
        fusion_type: "softmax_gate" | "sigmoid_gate".
        dropout:     classifier 직전 dropout rate.
    """

    def __init__(
        self,
        z_dim: int,
        num_sources: int,
        hidden_size: int,
        num_classes: int,
        fusion_type: str = "softmax_gate",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if fusion_type not in ("softmax_gate", "sigmoid_gate"):
            raise ValueError(f"fusion_type은 'softmax_gate' 또는 'sigmoid_gate'여야 합니다. 받은 값: '{fusion_type}'")

        self.gate = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_sources),
        )
        self.gate_act = nn.Softmax(dim=-1) if fusion_type == "softmax_gate" else nn.Sigmoid()
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(z_dim, num_classes)

    def forward(self, z_list: list[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Args:
            z_list: N개의 (B, z_dim) 텐서 리스트.

        Returns:
            logits:  (B, C) — 분류 logits
            weights: (B, N) — 각 소스에 대한 게이팅 가중치
        """
        stacked = torch.stack(z_list, dim=1)            # (B, N, z_dim)
        context = stacked.mean(dim=1)                    # (B, z_dim)
        weights = self.gate_act(self.gate(context))      # (B, N)
        z_fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (B, z_dim)
        logits = self.classifier(self.drop(z_fused))    # (B, C)
        return logits, weights


class StackingHead(nn.Module):
    """
    Logit 기반 앙상블 메타 학습기.

    각 모델의 raw logits를 softmax 확률로 변환 후 concat하여 MLP에 입력.
    z 특징보다 분류 신호가 직접적 → 동일 파라미터 수에서 더 높은 성능.

    Args:
        n_models:    입력 모델 수 (N).
        num_classes: 클래스 수 (C). 입력 차원 = N × C.
        hidden_size: MLP 내부 hidden 차원.
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        n_models: int,
        num_classes: int,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        in_dim = n_models * num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, logits_list: list[Tensor]) -> tuple[Tensor, None]:
        """
        Args:
            logits_list: N개의 (B, C) 텐서 리스트 — 각 모델의 raw logits.

        Returns:
            logits:  (B, C) — 앙상블 logits
            weights: None   (StackingHead는 per-sample weights 없음)
        """
        probs = torch.cat(
            [torch.softmax(l, dim=1) for l in logits_list], dim=1
        )  # (B, N*C)
        return self.net(probs), None
