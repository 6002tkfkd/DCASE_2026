"""
sk 전용 모델 정의.

dcase_ensemble에서 best-combo 4개 모델을 학습할 때 쓰인 EmbeddingEncoder는
hidden_size에 1024 캡이 걸려 있다 (MESH 공유 src/models/hatr.py는 캡이 없음):

    MESH (공유, 그대로 둠):            hidden_size = max(input_size, output_size * 2)
    dcase_ensemble (이 체크포인트들):  hidden_size = min(max(input_size, output_size * 2), 1024)

기존 체크포인트를 strict=True로 로드하려면 이 캡이 걸린 구조가 반드시 필요하다.
MESH의 공유 src/models/hatr.py는 수정하지 않고, 캡이 적용된 EmbeddingEncoder만
여기 따로 정의한다. ResidualBlock/AttentionFusion은 바뀐 게 없으므로 공유 모듈에서
그대로 가져다 쓴다.

best-combo 4개 모델의 체크포인트는 모두 config["model_name"]="BaseClassifier"로
저장되어 있다 (실험 디렉토리명이 "base_classifier_proxy_anchor_simple"이라 해서
실제 클래스가 proxy-anchor 계열인 것은 아님 — model.name은 출력 경로 라벨일 뿐,
실제 클래스는 training.run_mode로 결정됨). 특히 m2d 모델은 emb_size_audio=7680으로
캡이 실제로 적용되는 경우(min(7680,1024)=1024 vs max(7680,256)=7680)라 캡 버전이
필수다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.hatr import AttentionFusion, ResidualBlock


class EmbeddingEncoder(nn.Module):
    """src.models.hatr.EmbeddingEncoder와 동일하나 hidden_size가 1024로 캡된 버전."""

    def __init__(self, input_size, output_size, dropout=0.2, use_batch_norm=True, num_residual_blocks=3):
        super().__init__()

        hidden_size = min(max(input_size, output_size * 2), 1024)

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_size, hidden_size * 2, dropout, use_batch_norm)
                for _ in range(num_residual_blocks)
            ]
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.input_norm = nn.BatchNorm1d(input_size)
            self.output_norm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        if self.use_batch_norm:
            x = self.input_norm(x)

        x = self.input_projection(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.output_projection(x)

        if self.use_batch_norm:
            x = self.output_norm(x)

        return x


class BaseClassifier(nn.Module):
    """src.models.hatr.BaseClassifier와 동일하나 캡 적용 EmbeddingEncoder 사용."""

    def __init__(
        self,
        hidden_size=256,
        num_classes=10,
        emb_size_audio=0,
        emb_size_text=0,
        dropout=0.2,
        use_batch_norm=True,
        mode="both",
        num_residual_blocks=3,
        use_attention_fusion=True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb_size_audio = emb_size_audio
        self.emb_size_text = emb_size_text
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.mode = mode
        self.num_residual_blocks = num_residual_blocks
        self.use_attention_fusion = use_attention_fusion and mode == "both"

        if self.mode in ["audio", "both"]:
            self.audio_emb_extractor = EmbeddingEncoder(
                input_size=emb_size_audio,
                output_size=hidden_size,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                num_residual_blocks=num_residual_blocks,
            )
        else:
            self.audio_emb_extractor = None

        if self.mode in ["text", "both"]:
            self.text_emb_extractor = EmbeddingEncoder(
                input_size=emb_size_text,
                output_size=hidden_size,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                num_residual_blocks=num_residual_blocks,
            )
        else:
            self.text_emb_extractor = None

        if self.mode == "both":
            if self.use_attention_fusion:
                combined_size = hidden_size
                self.fusion = AttentionFusion(hidden_size, dropout)
            else:
                combined_size = hidden_size * 2
        else:
            combined_size = hidden_size

        self.latent_projector = nn.Sequential(
            nn.Linear(combined_size, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout / 2),
        )

        self.residual_classifier = nn.ModuleList(
            [
                ResidualBlock(hidden_size // 2, hidden_size, dropout / 2, use_batch_norm)
                for _ in range(2)
            ]
        )

        self.class_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout / 4),
            nn.Linear(hidden_size // 4, num_classes),
        )

    def forward(self, audio_emb=None, text_emb=None):
        features = []

        if self.mode in ["audio", "both"]:
            audio_features = self.audio_emb_extractor(audio_emb)
            features.append(audio_features)

        if self.mode in ["text", "both"]:
            text_features = self.text_emb_extractor(text_emb)
            features.append(text_features)

        if len(features) > 1:
            if self.use_attention_fusion:
                combined_features, attn_scores = self.fusion(features[0], features[1])
            else:
                combined_features = torch.cat(features, dim=-1)
                attn_scores = None
        else:
            combined_features = features[0]
            attn_scores = None

        z = self.latent_projector(combined_features)

        for block in self.residual_classifier:
            z = block(z)

        class_logit = self.class_predictor(z)

        return {
            "z": z,
            "logits": class_logit,
            "attn_scores": attn_scores,
        }


class BaseClassifierProxyAnchorSimple(nn.Module):
    """dcase_ensemble의 BaseClassifierProxyAnchorSimple과 동일 (캡된 EmbeddingEncoder 사용)."""

    def __init__(
        self,
        hidden_size=256,
        num_classes=10,
        emb_size_audio=0,
        emb_size_text=0,
        dropout=0.2,
        use_batch_norm=True,
        mode="both",
        num_residual_blocks=3,
        use_attention_fusion=True,
        embedding_dim=128,
        use_classifier=True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.emb_size_audio = emb_size_audio
        self.emb_size_text = emb_size_text
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.mode = mode
        self.num_residual_blocks = num_residual_blocks
        self.use_attention_fusion = use_attention_fusion and mode == "both"
        self.use_classifier = use_classifier

        if self.mode in ["audio", "both"]:
            self.audio_emb_extractor = EmbeddingEncoder(
                input_size=emb_size_audio,
                output_size=hidden_size,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                num_residual_blocks=num_residual_blocks,
            )
        else:
            self.audio_emb_extractor = None

        if self.mode in ["text", "both"]:
            self.text_emb_extractor = EmbeddingEncoder(
                input_size=emb_size_text,
                output_size=hidden_size,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                num_residual_blocks=num_residual_blocks,
            )
        else:
            self.text_emb_extractor = None

        if self.mode == "both":
            if self.use_attention_fusion:
                combined_size = hidden_size
                self.fusion = AttentionFusion(hidden_size, dropout)
            else:
                combined_size = hidden_size * 2
                self.fusion = None
        else:
            combined_size = hidden_size
            self.fusion = None

        self.latent_projector = nn.Linear(combined_size, embedding_dim)
        self.latent_dropout = nn.Dropout(dropout)

        self.class_predictor = (
            nn.Linear(embedding_dim, num_classes) if use_classifier else None
        )

    def forward(self, audio_emb=None, text_emb=None):
        features = []

        if self.mode in ["audio", "both"]:
            audio_features = self.audio_emb_extractor(audio_emb)
            features.append(audio_features)

        if self.mode in ["text", "both"]:
            text_features = self.text_emb_extractor(text_emb)
            features.append(text_features)

        if len(features) > 1:
            if self.use_attention_fusion and self.fusion is not None:
                combined_features, attn_scores = self.fusion(features[0], features[1])
            else:
                combined_features = torch.cat(features, dim=-1)
                attn_scores = None
        else:
            combined_features = features[0]
            attn_scores = None

        z = self.latent_projector(combined_features)
        z = self.latent_dropout(z)

        class_logit = self.class_predictor(z) if self.class_predictor is not None else None

        return {
            "z": z,
            "logits": class_logit,
            "attn_scores": attn_scores,
        }
