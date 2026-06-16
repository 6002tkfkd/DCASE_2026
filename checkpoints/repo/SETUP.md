# Checkpoint & Repo Setup

이 폴더에 아래 4개 repo를 clone하고, 각 checkpoint 폴더에 모델 가중치를 배치합니다.

---

## 1. Source Repos (git clone)

```bash
cd checkpoints/repo

# CLAP (LAION-AI)
git clone https://github.com/LAION-AI/CLAP

# PretrainedSED — fPaSST + ATST-Frame
git clone https://github.com/fschmid56/PretrainedSED

# microsoft/unilm — BEATs 소스코드 포함
git clone https://github.com/microsoft/unilm

# M2D
git clone https://github.com/nttcslab/m2d
```

Clone 후 폴더 구조:
```
checkpoints/repo/
  CLAP/src/            ← laion_clap 패키지
  PretrainedSED/       ← fPaSST, ATST 모델
  unilm/beats/         ← BEATs 모델 소스
  m2d/examples/        ← portable_m2d
```

### PaSST (pip install)

BEATs/CLAP/fPaSST/ATST/M2D는 repo clone으로 사용하지만,
PaSST는 pip 패키지로 제공됩니다:

```bash
pip install hear21passt
```

---

## 2. Model Weights

각 폴더에 아래 가중치 파일을 직접 다운로드/복사합니다.

### BEATs (`checkpoints/BEATs/`)

| 파일명 | 설명 |
|---|---|
| `BEATs_iter3_plus_AS2M.pt` | BEATs iter3+ AS2M (일반) |
| `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt` | BEATs iter3+ AS2M FT1 |

다운로드: https://github.com/microsoft/unilm/tree/master/beats

### CLAP 630K (`checkpoints/CLAP630/`)

| 파일명 | 설명 |
|---|---|
| `630k-audioset-fusion-best.pt` | CLAP 630K+AudioSet, Fusion |

다운로드: https://github.com/LAION-AI/CLAP

### CLAP Music (`checkpoints/CLAPM/`)

| 파일명 | 설명 |
|---|---|
| `music_audioset_epoch_15_esc_90.14.pt` | CLAP Music+AudioSet |

다운로드: https://github.com/LAION-AI/CLAP

### M2D (`checkpoints/M2D/`)

| 폴더/파일 | 설명 |
|---|---|
| `m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth` | M2D-CLAP ViT-Base |

다운로드: https://github.com/nttcslab/m2d

### PaSST (`checkpoints/PaSST/`)

PaSST는 `hear21passt` 패키지가 자동으로 다운로드합니다.
폴더만 준비되어 있으면 됩니다.

---

## 3. 준비 완료 확인

```bash
# 상태 확인
python scripts/extract_embeddings.py --paths paths.yaml --status

# 샘플 10개로 테스트
python scripts/extract_embeddings.py --paths paths.yaml \
    --dataset BSD10k-v1.2 --models beats_ft1 --limit 10
```

---

## 4. 전체 흐름

```bash
# Step 1: 임베딩 추출 (BSD10k + BSD35k)
python scripts/extract_embeddings.py --paths paths.yaml

# Step 2: per-model config 생성 (이미 있으면 생략)
python scripts/generate_embedding_configs.py

# Step 3: 전체 학습 (160 runs)
PATHS_FILE=paths.yaml SKIP_DONE=1 bash scripts/run_all_embedding_configs.sh

# Step 4: 앙상블 → System 1~4
python scripts/run_ensemble_experiments.py
python scripts/run_ensemble_search.py
```
