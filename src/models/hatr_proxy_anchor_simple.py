import torch
import torch.nn as nn

from .hatr import AttentionFusion, EmbeddingEncoder


class BaseClassifierProxyAnchorSimple(nn.Module):
    """Simplified proxy-anchor model with a shared linear embedding for both losses."""

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
