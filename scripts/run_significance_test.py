#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def _parse_confusions(cell: str):
    if not cell:
        return []
    obj = ast.literal_eval(cell)
    if isinstance(obj, list) and len(obj) == 2 and all(isinstance(x, list) for x in obj):
        return [obj]
    return obj if isinstance(obj, list) else []


def _binary_metrics_from_cm(cm: List[List[int]]) -> Dict[str, float]:
    tn, fp = float(cm[0][0]), float(cm[0][1])
    fn, tp = float(cm[1][0]), float(cm[1][1])
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0.0

    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0.0

    p0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0.0
    f1_macro = 0.5 * (f1_pos + f1_neg)
    return {"Accuracy": acc * 100.0, "F1_macro": f1_macro * 100.0, "F1_pos": f1_pos * 100.0}


def _load_seed_metrics(results_csv: Path, model_a: str, model_b: str):
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8", newline="")))
    out = {}
    for model in [model_a.lower(), model_b.lower()]:
        candidates = [
            r for r in rows
            if r.get("Model", "").strip().lower() == model and r.get("Confusion_Matrix", "").strip()
        ]
        if not candidates:
            raise ValueError(f"Model '{model}' with Confusion_Matrix not found in {results_csv}")
        idx = max(range(len(candidates)), key=lambda i: (len(_parse_confusions(candidates[i]["Confusion_Matrix"])), i))
        cms = _parse_confusions(candidates[idx]["Confusion_Matrix"])
        metrics = {"Accuracy": [], "F1_macro": [], "F1_pos": []}
        for cm in cms:
            m = _binary_metrics_from_cm(cm)
            for k in metrics:
                metrics[k].append(m[k])
        out[model] = {k: np.asarray(v, dtype=np.float64) for k, v in metrics.items()}
    return out


def _holm_bonferroni(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def _paired_permutation_pvalue(diff: np.ndarray, num_permutations: int, rng: np.random.Generator) -> float:
    obs = float(np.abs(np.mean(diff)))
    if diff.size == 0:
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(num_permutations, diff.size))
    perm_means = np.abs((signs * diff).mean(axis=1))
    p = (np.sum(perm_means >= obs) + 1.0) / (num_permutations + 1.0)
    return float(p)


def _bootstrap_ci(diff: np.ndarray, num_bootstrap: int, rng: np.random.Generator):
    if diff.size == 0:
        return 0.0, 0.0
    idx = rng.integers(0, diff.size, size=(num_bootstrap, diff.size))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


@dataclass
class TestResult:
    p_two_sided: float
    p_holm: float = 1.0


def main():
    parser = argparse.ArgumentParser(description="Compute paired significance tests from seed-level confusion matrices.")
    parser.add_argument("--results-csv", type=Path, required=True, help="CSV containing Model and Confusion_Matrix columns.")
    parser.add_argument("--model-a", type=str, default="camfn", help="Primary model name in CSV.")
    parser.add_argument("--model-b", type=str, default="muvac", help="Baseline model name in CSV.")
    parser.add_argument("--num-permutations", type=int, default=50000, help="Permutation test iterations.")
    parser.add_argument("--num-bootstrap", type=int, default=50000, help="Bootstrap iterations for CI.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for permutation/bootstrap.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file path.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    models = _load_seed_metrics(args.results_csv, args.model_a, args.model_b)
    a = models[args.model_a.lower()]
    b = models[args.model_b.lower()]

    metrics = ["Accuracy", "F1_macro", "F1_pos"]
    results = {}
    for metric in metrics:
        xa = a[metric]
        xb = b[metric]
        if xa.shape != xb.shape:
            raise ValueError(f"Seed count mismatch for {metric}: {xa.shape} vs {xb.shape}")
        diff = xa - xb
        if diff.size < 2:
            raise ValueError(f"Need at least 2 paired runs for significance testing, got {diff.size}")

        ttest = ttest_rel(xa, xb, alternative="two-sided")
        wil = wilcoxon(diff, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
        p_perm = _paired_permutation_pvalue(diff, args.num_permutations, rng)
        ci_low, ci_high = _bootstrap_ci(diff, args.num_bootstrap, rng)

        results[metric] = {
            "n_pairs": int(diff.size),
            "mean_a": float(np.mean(xa)),
            "mean_b": float(np.mean(xb)),
            "mean_diff_a_minus_b": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "paired_t": TestResult(p_two_sided=float(ttest.pvalue)).__dict__,
            "permutation": TestResult(p_two_sided=float(p_perm)).__dict__,
            "wilcoxon": TestResult(p_two_sided=float(wil.pvalue)).__dict__,
            "bootstrap": {"ci95_low": ci_low, "ci95_high": ci_high},
        }

    for test_name in ["paired_t", "permutation", "wilcoxon"]:
        raw = [results[m][test_name]["p_two_sided"] for m in metrics]
        holm = _holm_bonferroni(raw)
        for i, metric in enumerate(metrics):
            results[metric][test_name]["p_holm"] = holm[i]

    output = {
        "meta": {
            "results_csv": str(args.results_csv),
            "model_a": args.model_a,
            "model_b": args.model_b,
            "num_permutations": args.num_permutations,
            "num_bootstrap": args.num_bootstrap,
            "seed": args.seed,
        },
        "tests": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved significance report to: {args.output}")


if __name__ == "__main__":
    main()
