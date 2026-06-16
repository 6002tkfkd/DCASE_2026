#!/usr/bin/env bash
# Run all 270 embedding configs: 9 models × 2 exp_dirs × 15 configs each
# Usage:
#   bash scripts/run_all_embedding_configs.sh                          # 전체 실행
#   DRY_RUN=1 bash scripts/run_all_embedding_configs.sh               # 명령어만 출력
#   SKIP_DONE=1 bash scripts/run_all_embedding_configs.sh             # 이미 완료된 config 건너뜀
#   MODEL=baseline bash scripts/run_all_embedding_configs.sh          # 특정 모델만
#   NO_TSNE=1 bash scripts/run_all_embedding_configs.sh               # tsne 생략
#   PATHS_FILE=paths.yaml bash scripts/run_all_embedding_configs.sh   # 경로 override

set -euo pipefail

EXP_DIRS=(
  "configs/two_stage_exp1_hloss"
  "configs/two_stage_exp2_proxy"
)

MODELS=(
  "baseline"
  "atst_weak_seq_maxstd_chunk_meanstd"
  "beats_iter3plus_as2m_ft1_seq_meanstd_chunk_mean"
  "beats_iter3plus_as2m_seq_meanmaxstd_chunk_max"
  "clap_630k_chunk_meanmax"
  "clap_music_as_chunk_mean"
  "fpasst_weak_seq_meanmax_chunk_maxstd"
  "m2d_clap_vit_base_meanstd"
  "passt_kd_ap486_chunk_max"
)

LOG_DIR="${LOG_DIR:-output/embedding_run_logs}"
FILTER_MODEL="${MODEL:-}"       # 특정 모델만 돌릴 때 MODEL=xxx 설정
PATHS_ARG=""
if [ -n "${PATHS_FILE-}" ]; then
  PATHS_ARG="--paths ${PATHS_FILE}"
fi

if [ -z "${DRY_RUN-}" ]; then
  mkdir -p "$LOG_DIR"
fi

total=0
skipped=0
failed=0
succeeded=0

for exp_dir in "${EXP_DIRS[@]}"; do
  exp_name="$(basename "$exp_dir")"

  for model in "${MODELS[@]}"; do
    # MODEL 필터
    if [ -n "$FILTER_MODEL" ] && [ "$model" != "$FILTER_MODEL" ]; then
      continue
    fi

    model_dir="${exp_dir}/${model}"
    if [ ! -d "$model_dir" ]; then
      echo "[WARN] 디렉토리 없음: $model_dir"
      continue
    fi

    for cfg in "$model_dir"/*.yaml; do
      [ -f "$cfg" ] || continue
      total=$((total + 1))

      name="$(basename "$cfg" .yaml)"
      log_prefix="${exp_name}_${model}_${name}"

      # SKIP_DONE: run_dir 하위에 결과 파일 존재 여부로 판단
      if [ -n "${SKIP_DONE-}" ]; then
        # summary_results.txt 존재 여부로 완료 판단
        result_dir="output/${model}/${exp_name}/${name}"
        if [ -f "$result_dir/base_classifier_proxy_anchor_simple/both/summary_results.txt" ]; then
          skipped=$((skipped + 1))
          echo "[SKIP] $log_prefix"
          continue
        fi
      fi

      echo ""
      echo "=== [${exp_name}] [${model}] ${name} ==="

      if [ -n "${DRY_RUN-}" ]; then
        echo "  DRY: python scripts/train_twostage.py --config \"$cfg\" ${PATHS_ARG}"
        if [ -z "${NO_TSNE-}" ]; then
          echo "  DRY: python scripts/generate_tsne.py --config \"$cfg\" --stages pretrain,finetune"
        fi
        continue
      fi

      # 학습
      # shellcheck disable=SC2086
      if python scripts/train_twostage.py --config "$cfg" ${PATHS_ARG} 2>&1 | tee "$LOG_DIR/${log_prefix}_train.log"; then
        succeeded=$((succeeded + 1))
      else
        failed=$((failed + 1))
        echo "[FAILED] $log_prefix" | tee -a "$LOG_DIR/failed.txt"
        continue
      fi

      # 평가 완료 후 pth 삭제 (결과값만 보존, 용량 절약)
      pth_dir="output/${model}/${exp_name}/${name}"
      pth_count=$(find "$pth_dir" -name "*.pth" 2>/dev/null | wc -l)
      if [ "$pth_count" -gt 0 ]; then
        find "$pth_dir" -name "*.pth" -delete
        echo "  [CLEANUP] .pth 파일 ${pth_count}개 삭제: $pth_dir"
      fi

      # tsne (선택)
      if [ -z "${NO_TSNE-}" ]; then
        python scripts/generate_tsne.py --config "$cfg" --stages pretrain,finetune \
          2>&1 | tee "$LOG_DIR/${log_prefix}_tsne.log" || true
      fi
    done
  done
done

echo ""
echo "========================================"
echo "전체: ${total}  완료: ${succeeded}  건너뜀: ${skipped}  실패: ${failed}"
echo "========================================"
