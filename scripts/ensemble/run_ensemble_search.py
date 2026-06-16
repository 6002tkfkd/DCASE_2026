#!/usr/bin/env python3
"""Ensemble search: greedy subset selection + continuous weight optimization.

Part A — Weight optimization (scipy multi-start):
  주어진 입력 집합에 대해 softmax-parameterized 연속 가중치를 최적화한다.
  이산 grid {1,2,3,4} 탐색 대신 [0,1] 연속 공간에서 50회 random-restart minimize.

Part B — Greedy forward selection:
  캐시된 모든 run 중 가장 높은 hF1을 내는 조합을 탐욕적으로 구성한다.
  각 스텝에서 현재 집합에 하나씩 추가하고, 추가 후 weight optimization 수행.

사용법:
  # A: 기존 ensemble 입력에 대해 가중치만 최적화
  python scripts/run_ensemble_search.py --mode weights \\
      --inputs output_ensemble/cross_top1/uniform/inputs.csv

  # B: 전체 캐시 pool에서 greedy subset 탐색
  python scripts/run_ensemble_search.py --mode greedy --max-k 12

  # A+B 둘 다
  python scripts/run_ensemble_search.py --mode all --max-k 12
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.optimize import minimize

import scripts.run_ensemble_experiments as ens


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_all_specs_from_cache(cache_dir: Path) -> List[ens.RunSpec]:
    """cache/oof_softmax/<model>/<exp>/<config>/oof_softmax.npz 에서 spec 자동 탐색."""
    specs = []
    for npz in sorted(cache_dir.glob("*/*/*/oof_softmax.npz")):
        meta_path = npz.with_suffix(".json")
        if meta_path.exists():
            with meta_path.open() as f:
                m = json.load(f)
            specs.append(ens.RunSpec(
                model=m["model"], exp=m["exp"], config=m["config"],
                rank=m.get("rank", 99), hf1=m.get("hf1", 0.0),
            ))
        else:
            parts = npz.parts
            # .../cache/oof_softmax/<model>/<exp>/<config>/oof_softmax.npz
            config = parts[-2]
            exp    = parts[-3]
            model  = parts[-4]
            specs.append(ens.RunSpec(model=model, exp=exp, config=config, rank=99, hf1=0.0))
    return specs


def load_aligned_from_specs(
    output_dir: Path,
    specs: Sequence[ens.RunSpec],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """load_cache → 공통 ID 교집합 → (ids, probs_stack[K,N,C], labels[N], id2class)."""
    per_run = []
    common_ids = None
    id2class_ref = None
    for spec in specs:
        ids, probs, labels, id2class = ens.load_cache(output_dir, spec)
        mapping = {sid: i for i, sid in enumerate(ids.tolist())}
        per_run.append((mapping, probs, labels, id2class))
        ids_set = set(mapping.keys())
        common_ids = ids_set if common_ids is None else common_ids & ids_set
        if id2class_ref is None:
            id2class_ref = id2class

    ordered_ids = sorted(common_ids, key=lambda x: int(x) if x.isdigit() else x)
    probs_list, labels_ref = [], None
    for mapping, probs, labels, _ in per_run:
        idx = np.asarray([mapping[sid] for sid in ordered_ids], dtype=np.int64)
        probs_list.append(probs[idx].astype(np.float32))
        if labels_ref is None:
            labels_ref = labels[idx].astype(np.int64)
    return (
        np.asarray(ordered_ids, dtype=str),
        np.stack(probs_list, axis=0),   # (K, N, C)
        labels_ref,
        id2class_ref,
    )


def compute_hf1_from_stack(probs_stack: np.ndarray, weights: np.ndarray,
                            labels: np.ndarray, id2class: Dict[int, str]) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    combined = np.tensordot(w, probs_stack.astype(np.float64), axes=(0, 0)).astype(np.float32)
    return ens.compute_metrics(labels, combined, id2class)["hierarchical_f1"]


# ---------------------------------------------------------------------------
# Part A: Continuous weight optimization (scipy multi-start)
# ---------------------------------------------------------------------------

def softmax(u: np.ndarray) -> np.ndarray:
    e = np.exp(u - u.max())
    return e / e.sum()


def compute_accuracy_from_stack(probs_stack: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> float:
    """빠른 proxy metric: 단순 accuracy (numpy only, Python loop 없음)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    combined = np.tensordot(w, probs_stack.astype(np.float64), axes=(0, 0))
    preds = combined.argmax(axis=1)
    return float(np.mean(preds == labels)) * 100.0


def optimize_weights(
    probs_stack: np.ndarray,
    labels: np.ndarray,
    id2class: Dict[int, str],
    n_restarts: int = 50,
    regularization: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """
    탐색: accuracy로 빠르게 최적화 (scipy 수천 호출 대응)
    검증: 상위 후보에만 hF1 계산
    """
    K = probs_stack.shape[0]
    uniform = np.ones(K) / K

    def neg_acc(u):
        w = softmax(u)
        acc = compute_accuracy_from_stack(probs_stack, w, labels)
        penalty = regularization * float(np.sum((w - uniform) ** 2))
        return -(acc - 100.0 * penalty)

    # accuracy 기준 multi-start 최적화
    candidates = []
    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        u0 = rng.normal(0, 0.5, K)
        res = minimize(neg_acc, u0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5})
        w = softmax(res.x)
        acc = compute_accuracy_from_stack(probs_stack, w, labels)
        candidates.append((acc, w))

    # 상위 5개 후보에 대해서만 hF1 계산
    candidates.sort(key=lambda x: -x[0])
    best_w, best_hf1 = uniform, compute_hf1_from_stack(probs_stack, uniform, labels, id2class)
    for _, w in candidates[:5]:
        hf1 = compute_hf1_from_stack(probs_stack, w, labels, id2class)
        if hf1 > best_hf1:
            best_hf1, best_w = hf1, w

    return best_w, best_hf1


def run_weight_search(
    output_dir: Path,
    inputs_csv: Path,
    out_dir: Path,
    n_restarts: int,
    regularization: float,
):
    specs = []
    with inputs_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            specs.append(ens.RunSpec(
                model=row["model"], exp=row["exp"], config=row["config"],
                rank=int(row.get("rank", 99)), hf1=float(row.get("hf1", 0)),
            ))

    print(f"\n[WeightSearch] {inputs_csv.parent.name}  inputs={len(specs)}")
    sound_ids, probs_stack, labels, id2class = load_aligned_from_specs(output_dir, specs)
    print(f"  samples={len(sound_ids)}  inputs={probs_stack.shape[0]}")

    uniform_hf1 = compute_hf1_from_stack(probs_stack, np.ones(len(specs)), labels, id2class)
    print(f"  uniform hF1 : {uniform_hf1:.4f}%")

    t0 = time.time()
    best_w, best_hf1 = optimize_weights(probs_stack, labels, id2class, n_restarts, regularization)
    elapsed = time.time() - t0

    print(f"  best   hF1 : {best_hf1:.4f}%  (+{best_hf1 - uniform_hf1:.4f}%)  [{elapsed:.1f}s, {n_restarts} restarts]")
    print(f"  weights     : {[round(float(w), 4) for w in best_w]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_w = np.tensordot(best_w.astype(np.float64), probs_stack.astype(np.float64), axes=(0, 0)).astype(np.float32)
    metrics = ens.compute_metrics(labels, combined_w, id2class)

    ens.write_inputs_csv(out_dir / "inputs.csv", specs, best_w.tolist())
    ens.write_predictions_csv(out_dir / "predictions.csv", sound_ids, labels, combined_w, id2class)
    np.savez_compressed(out_dir / "oof_softmax.npz", sound_ids=sound_ids, probs=combined_w, labels=labels)

    result = {
        "source_inputs": str(inputs_csv),
        "search_mode": "scipy_multistart",
        "n_restarts": n_restarts,
        "regularization": regularization,
        "num_inputs": len(specs),
        "uniform_hf1": uniform_hf1,
        "best_hf1": best_hf1,
        "best_weights": [float(w) for w in best_w],
        "metrics": metrics,
    }
    with (out_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Part B: Greedy forward selection
# ---------------------------------------------------------------------------

def greedy_forward_select(
    output_dir: Path,
    pool_specs: List[ens.RunSpec],
    max_k: int,
    n_restarts: int,
    regularization: float,
    out_dir: Path,
    optimize_each_step: bool = True,
) -> List[dict]:
    """
    가장 높은 hF1을 내는 run을 탐욕적으로 추가한다.
    각 스텝에서 나머지 모든 후보를 평가 후 최적을 선택.
    optimize_each_step=True: 선택 후 가중치 최적화.
    """
    print(f"\n[GreedySearch] pool={len(pool_specs)}  max_k={max_k}")

    # 전체 pool 캐시를 한 번에 로드
    print("  Loading all cached runs...")
    t0 = time.time()
    all_ids_list, all_probs_list, labels_ref, id2class = [], [], None, None
    valid_specs = []
    for spec in pool_specs:
        try:
            ids, probs, labels, i2c = ens.load_cache(output_dir, spec)
        except Exception as e:
            print(f"  [SKIP] {spec.key}: {e}")
            continue
        all_ids_list.append(ids)
        all_probs_list.append(probs)
        if labels_ref is None:
            labels_ref, id2class = labels, i2c
        valid_specs.append(spec)
    print(f"  Loaded {len(valid_specs)} runs in {time.time()-t0:.1f}s")

    # 공통 ID 교집합
    common_ids = None
    for ids in all_ids_list:
        s = set(ids.astype(str).tolist())
        common_ids = s if common_ids is None else common_ids & s
    ordered_ids = sorted(common_ids, key=lambda x: int(x) if x.isdigit() else x)
    N = len(ordered_ids)

    # 공통 ID 기준으로 probs 재정렬
    pool_probs = []  # list of (N, C) arrays
    pool_labels = None
    for ids, probs, labels, _ in zip(all_ids_list, all_probs_list, [labels_ref]*len(all_ids_list), [None]*len(all_ids_list)):
        mapping = {sid: i for i, sid in enumerate(ids.astype(str).tolist())}
        idx = np.asarray([mapping[sid] for sid in ordered_ids], dtype=np.int64)
        pool_probs.append(probs[idx].astype(np.float32))
        if pool_labels is None:
            lids, lprobs, llabels, _ = ens.load_cache(output_dir, valid_specs[0])
            lmap = {sid: i for i, sid in enumerate(lids.astype(str).tolist())}
            lidx = np.asarray([lmap[sid] for sid in ordered_ids], dtype=np.int64)
            pool_labels = llabels[lidx].astype(np.int64)

    # labels 재로드 (위 방법이 복잡하므로 직접)
    first_ids, first_probs, first_labels, id2class = ens.load_cache(output_dir, valid_specs[0])
    first_map = {sid: i for i, sid in enumerate(first_ids.astype(str).tolist())}
    first_idx = np.asarray([first_map[sid] for sid in ordered_ids], dtype=np.int64)
    pool_labels = first_labels[first_idx].astype(np.int64)

    print(f"  Common samples: {N}")

    selected_indices = []   # pool_probs의 인덱스
    selected_specs   = []
    step_results     = []
    remaining_idx    = list(range(len(valid_specs)))

    for step in range(1, max_k + 1):
        best_candidate_idx = None
        best_hf1_step = -1.0

        # 현재 선택 + 각 후보를 추가해서 uniform 평가
        for cand_i in remaining_idx:
            trial_indices = selected_indices + [cand_i]
            trial_stack   = np.stack([pool_probs[i] for i in trial_indices], axis=0)
            hf1 = compute_hf1_from_stack(trial_stack, np.ones(len(trial_indices)), pool_labels, id2class)
            if hf1 > best_hf1_step:
                best_hf1_step      = hf1
                best_candidate_idx = cand_i

        selected_indices.append(best_candidate_idx)
        selected_specs.append(valid_specs[best_candidate_idx])
        remaining_idx.remove(best_candidate_idx)

        # 현재 선택 집합으로 가중치 최적화
        current_stack = np.stack([pool_probs[i] for i in selected_indices], axis=0)
        if optimize_each_step and step > 1:
            best_w, opt_hf1 = optimize_weights(current_stack, pool_labels, id2class,
                                               n_restarts=min(n_restarts, 30), regularization=regularization)
        else:
            best_w   = np.ones(step) / step
            opt_hf1  = compute_hf1_from_stack(current_stack, best_w, pool_labels, id2class)

        added = valid_specs[best_candidate_idx]
        print(f"  step {step:2d}: added={added.model[:25]}/{added.config[:30]}  "
              f"uniform={best_hf1_step:.4f}%  opt={opt_hf1:.4f}%")

        step_results.append({
            "step": step,
            "added_key": added.key,
            "model": added.model,
            "exp": added.exp,
            "config": added.config,
            "individual_hf1": added.hf1,
            "uniform_hf1": best_hf1_step,
            "opt_hf1": opt_hf1,
            "weights": [float(w) for w in best_w],
            "selected_keys": [valid_specs[i].key for i in selected_indices],
        })

        # 결과 저장 (스텝별)
        step_dir = out_dir / f"step_{step:02d}_{added.model[:20]}"
        step_dir.mkdir(parents=True, exist_ok=True)
        combined_w = np.tensordot(best_w.astype(np.float64), current_stack.astype(np.float64), axes=(0, 0)).astype(np.float32)
        metrics = ens.compute_metrics(pool_labels, combined_w, id2class)
        ens.write_inputs_csv(step_dir / "inputs.csv", selected_specs, best_w.tolist())
        with (step_dir / "result.json").open("w") as f:
            json.dump({"step": step, "metrics": metrics, "weights": [float(w) for w in best_w],
                       "selected": [s.key for s in selected_specs]}, f, indent=2)

    return step_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_greedy_summary(out_dir: Path, step_results: List[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "greedy_summary.csv"
    fields = ["step", "model", "config", "individual_hf1", "uniform_hf1", "opt_hf1"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in step_results:
            writer.writerow({k: row[k] for k in fields})

    json_path = out_dir / "greedy_summary.json"
    with json_path.open("w") as f:
        json.dump(step_results, f, indent=2)
    print(f"\n[SUMMARY] {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_same_model_greedy(
    output_dir: Path,
    cache_dir: Path,
    out_dir: Path,
    max_k: int,
    n_restarts: int,
    regularization: float,
) -> List[dict]:
    """각 모델별로 1→max_k 개씩 greedy forward selection + 가중치 최적화."""
    pool_specs = load_all_specs_from_cache(cache_dir)
    from collections import defaultdict
    by_model: Dict[str, List[ens.RunSpec]] = defaultdict(list)
    for spec in pool_specs:
        by_model[spec.model].append(spec)

    summary_rows = []
    best_per_model = {}

    for model, specs in sorted(by_model.items()):
        n_pool = len(specs)
        k = min(max_k, n_pool)
        print(f"\n[SameModel] {model}  pool={n_pool}  max_k={k}")

        # 모델의 모든 캐시 로드
        all_ids_list, all_probs_list = [], []
        valid_specs = []
        for spec in specs:
            try:
                ids, probs, labels, id2class = ens.load_cache(output_dir, spec)
            except Exception as e:
                print(f"  [SKIP] {spec.config}: {e}")
                continue
            all_ids_list.append(ids)
            all_probs_list.append(probs)
            valid_specs.append(spec)

        if not valid_specs:
            continue

        # 공통 ID 교집합
        common_ids = None
        for ids in all_ids_list:
            s = set(ids.astype(str).tolist())
            common_ids = s if common_ids is None else common_ids & s
        ordered_ids = sorted(common_ids, key=lambda x: int(x) if x.isdigit() else x)

        first_ids, _, first_labels, id2class = ens.load_cache(output_dir, valid_specs[0])
        first_map = {sid: i for i, sid in enumerate(first_ids.astype(str).tolist())}
        first_idx = np.asarray([first_map[sid] for sid in ordered_ids], dtype=np.int64)
        pool_labels = first_labels[first_idx].astype(np.int64)

        pool_probs = []
        for ids, probs in zip(all_ids_list, all_probs_list):
            mapping = {sid: i for i, sid in enumerate(ids.astype(str).tolist())}
            idx = np.asarray([mapping[sid] for sid in ordered_ids], dtype=np.int64)
            pool_probs.append(probs[idx].astype(np.float32))

        selected_indices = []
        remaining_idx = list(range(len(valid_specs)))
        model_best_hf1 = -1.0  # opt_hf1 기준
        model_best_step = 0

        for step in range(1, k + 1):
            # greedy: uniform으로 candidate 평가 (빠름)
            best_cand_i = None
            best_uniform_hf1 = -1.0
            for cand_i in remaining_idx:
                trial_indices = selected_indices + [cand_i]
                trial_stack = np.stack([pool_probs[i] for i in trial_indices], axis=0)
                hf1 = compute_hf1_from_stack(trial_stack, np.ones(len(trial_indices)), pool_labels, id2class)
                if hf1 > best_uniform_hf1:
                    best_uniform_hf1 = hf1
                    best_cand_i = cand_i

            selected_indices.append(best_cand_i)
            remaining_idx.remove(best_cand_i)
            added = valid_specs[best_cand_i]

            # 선택된 집합에 가중치 최적화
            current_stack = np.stack([pool_probs[i] for i in selected_indices], axis=0)
            if step > 1:
                best_w, opt_hf1 = optimize_weights(
                    current_stack, pool_labels, id2class,
                    n_restarts=min(n_restarts, 30),
                    regularization=regularization,
                )
            else:
                best_w = np.ones(1)
                opt_hf1 = best_uniform_hf1

            print(f"  step {step:2d}: {added.config[:40]}  uniform={best_uniform_hf1:.4f}%  opt={opt_hf1:.4f}%")

            row = {
                "model": model,
                "step": step,
                "config": added.config,
                "uniform_hf1": round(best_uniform_hf1, 4),
                "opt_hf1": round(opt_hf1, 4),
                "weights": [round(float(w), 4) for w in best_w],
                "is_best": False,
            }
            summary_rows.append(row)

            if opt_hf1 > model_best_hf1:
                model_best_hf1 = opt_hf1
                model_best_step = step

        best_per_model[model] = {"step": model_best_step, "hf1": model_best_hf1}
        print(f"  => BEST: step={model_best_step}  opt_hF1={model_best_hf1:.4f}%")

    # best 표시
    for row in summary_rows:
        if best_per_model[row["model"]]["step"] == row["step"]:
            row["is_best"] = True

    # CSV 저장
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "same_model_greedy_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "step", "config", "uniform_hf1", "opt_hf1", "is_best"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row[k] for k in ["model", "step", "config", "uniform_hf1", "opt_hf1", "is_best"]})
    print(f"\n[SAVED] {csv_path}")

    # 모델별 best 요약 출력
    print("\n" + "="*60)
    print("Per-Model Best (same-model greedy + weight opt)")
    print("="*60)
    print(f"{'model':<45}  {'best_step':>9}  {'opt_hF1':>9}")
    print("-"*65)
    sorted_models = sorted(best_per_model.items(), key=lambda x: -x[1]["hf1"])
    for model, info in sorted_models:
        print(f"  {model:<43}  {info['step']:>9}  {info['hf1']:>9.4f}%")

    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Greedy subset + continuous weight search for ensemble")
    parser.add_argument("--mode", choices=["weights", "greedy", "all", "same_model"], default="all")
    parser.add_argument("--output-dir",   default=str(PROJECT_ROOT / "output_ensemble"))
    parser.add_argument("--search-dir",   default=str(PROJECT_ROOT / "output_ensemble_search"))
    parser.add_argument("--inputs",       default=None,
                        help="[weights mode] inputs.csv 경로 (기본: cross_top1/uniform/inputs.csv)")
    parser.add_argument("--max-k",        type=int, default=12,
                        help="[greedy mode] 최대 선택 모델 수")
    parser.add_argument("--n-restarts",   type=int, default=50,
                        help="scipy multi-start 횟수")
    parser.add_argument("--regularization", type=float, default=0.05,
                        help="uniform 방향 L2 패널티 (0=없음, 클수록 uniform에 가까워짐)")
    parser.add_argument("--no-step-opt",  action="store_true",
                        help="greedy 각 스텝에서 weight 최적화 생략 (빠르게 탐색만)")
    parser.add_argument("--pool-top-n",   type=int, default=0,
                        help="greedy pool을 각 모델의 top-N으로 제한 (0=전체 캐시 사용)")
    parser.add_argument("--same-model-out", default=None,
                        help="[same_model mode] 결과 저장 경로 (기본: output_ensemble_search/same_model_greedy)")
    parser.add_argument("--pool-filter-exp", default=None,
                        help="[greedy mode] exp 이름으로 pool 필터 (예: two_stage_exp1_hloss)")
    parser.add_argument("--greedy-out", default=None,
                        help="[greedy mode] 결과 저장 경로 (기본: output_ensemble_search/greedy)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    search_dir = Path(args.search_dir).resolve()
    cache_dir  = output_dir / "cache" / "oof_softmax"

    # ---------- Part A: Weight optimization ----------
    if args.mode in ("weights", "all"):
        if args.inputs:
            p = Path(args.inputs)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            inputs_csvs = [p]
        else:
            # 기본: output_ensemble 내 모든 inputs.csv
            inputs_csvs = sorted(output_dir.glob("**/inputs.csv"))
            # cache/ 하위 제외
            inputs_csvs = [p for p in inputs_csvs if "cache" not in p.parts]

        weight_results = []
        for inputs_csv in inputs_csvs:
            if not inputs_csv.exists():
                continue
            rel = inputs_csv.parent.relative_to(output_dir)
            out = search_dir / "weight_opt" / rel
            r = run_weight_search(output_dir, inputs_csv, out,
                                  args.n_restarts, args.regularization)
            weight_results.append(r)

        # weight search 요약
        if weight_results:
            print("\n" + "="*60)
            print("Weight Optimization Summary")
            print("="*60)
            weight_results.sort(key=lambda x: x["best_hf1"], reverse=True)
            for r in weight_results:
                src = Path(r["source_inputs"]).parent.relative_to(output_dir)
                print(f"  {str(src):<40} uniform={r['uniform_hf1']:.4f}%  best={r['best_hf1']:.4f}%")

            summary_path = search_dir / "weight_opt_summary.csv"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["source", "n_inputs", "uniform_hf1", "best_hf1", "gain"])
                writer.writeheader()
                for r in weight_results:
                    src = Path(r["source_inputs"]).parent.relative_to(output_dir)
                    writer.writerow({
                        "source": str(src),
                        "n_inputs": r["num_inputs"],
                        "uniform_hf1": round(r["uniform_hf1"], 4),
                        "best_hf1": round(r["best_hf1"], 4),
                        "gain": round(r["best_hf1"] - r["uniform_hf1"], 4),
                    })
            print(f"\n[SAVED] {summary_path}")

    # ---------- Part C: Same-model greedy ----------
    if args.mode == "same_model":
        same_model_out = Path(args.same_model_out).resolve() if args.same_model_out else search_dir / "same_model_greedy"
        run_same_model_greedy(
            output_dir=output_dir,
            cache_dir=cache_dir,
            out_dir=same_model_out,
            max_k=args.max_k,
            n_restarts=args.n_restarts,
            regularization=args.regularization,
        )
        return

    # ---------- Part B: Greedy selection ----------
    if args.mode in ("greedy", "all"):
        pool_specs = load_all_specs_from_cache(cache_dir)
        print(f"\n[POOL] {len(pool_specs)} cached runs found")

        # pool_filter_exp: 특정 experiment만 사용 (e.g. two_stage_exp1_hloss)
        if args.pool_filter_exp:
            pool_specs = [s for s in pool_specs if s.exp == args.pool_filter_exp]
            print(f"[POOL] filtered to exp={args.pool_filter_exp}: {len(pool_specs)} runs")

        # pool_top_n: 각 모델의 상위 N개만 사용
        if args.pool_top_n > 0:
            from collections import defaultdict
            by_model: Dict[str, List] = defaultdict(list)
            for spec in pool_specs:
                by_model[spec.model].append(spec)
            pool_specs = []
            for model, specs in by_model.items():
                specs.sort(key=lambda s: (-s.hf1, s.rank))
                pool_specs.extend(specs[:args.pool_top_n])
            print(f"[POOL] filtered to top-{args.pool_top_n} per model: {len(pool_specs)} runs")

        exp_tag = f"_{args.pool_filter_exp}" if args.pool_filter_exp else ""
        greedy_out = Path(args.greedy_out).resolve() if args.greedy_out else search_dir / f"greedy{exp_tag}"
        step_results = greedy_forward_select(
            output_dir=output_dir,
            pool_specs=pool_specs,
            max_k=args.max_k,
            n_restarts=args.n_restarts,
            regularization=args.regularization,
            out_dir=greedy_out,
            optimize_each_step=not args.no_step_opt,
        )
        write_greedy_summary(greedy_out, step_results)

        print("\n" + "="*60)
        print("Greedy Selection Summary")
        print("="*60)
        print(f"{'step':>4}  {'model':<28}  {'uniform':>9}  {'opt':>9}")
        print("-"*60)
        for r in step_results:
            print(f"  {r['step']:>2}  {r['model'][:28]:<28}  {r['uniform_hf1']:>9.4f}%  {r['opt_hf1']:>9.4f}%")


if __name__ == "__main__":
    main()
