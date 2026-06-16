"""
sk 경로 헬퍼.

dcase_ensemble에서 이미 학습된 체크포인트/임베딩을 심링크 없이 상대경로
(`legacy_data_root`, 보통 "../dcase_ensemble")로 참조하기 위한 유틸리티.

processed_dataset.csv의 audio_emb_filepath/text_emb_filepath 컬럼에는
dcase_ensemble 저장소 루트 기준 상대경로 문자열(예: "./embedding/.../185755.npy")이
그대로 박혀 있다. legacy_data_root를 그 앞에 붙여서 풀어준다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_EMBEDDING_PATH_COLUMNS = ("audio_emb_filepath", "text_emb_filepath")


def resolve_legacy(rel_path: str, legacy_root: str) -> Path:
    """legacy_root 기준 상대경로를 결합해 반환."""
    return Path(legacy_root) / rel_path


def rewrite_embedding_columns(df: pd.DataFrame, legacy_root: str) -> pd.DataFrame:
    """
    processed_dataset.csv(또는 splits 병합 df)의 임베딩 경로 컬럼 앞에
    legacy_root를 붙여서 MESH 쪽 CWD에서도 올바르게 resolve되도록 교체.

    원본 df는 변경하지 않고 복사본을 반환.
    """
    df = df.copy()
    for col in _EMBEDDING_PATH_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(lambda p: str(resolve_legacy(p, legacy_root)) if pd.notna(p) else p)
    return df
