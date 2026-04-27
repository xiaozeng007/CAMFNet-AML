#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def _load_seed_metrics_from_csv(results_csv: Path) -> Dict[str, Dict[str, np.ndarray]]:
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8", newline="")))
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for model in ["camfn", "muvac"]:
        cands = [r for r in rows if (r.get("Model", "").strip().lower() == model and r.get("Confusion_Matrix", "").strip())]
        if not cands:
            continue
        # choose row with most seeds, latest on tie
        idx = max(range(len(cands)), key=lambda i: (len(_parse_confusions(cands[i]["Confusion_Matrix"])), i))
        row = cands[idx]
        cms = _parse_confusions(row["Confusion_Matrix"])
        a, f1m, f1p = [], [], []
        for cm in cms:
            m = _binary_metrics_from_cm(cm)
            a.append(m["Accuracy"])
            f1m.append(m["F1_macro"])
            f1p.append(m["F1_pos"])
        out[model] = {
            "Accuracy": np.array(a, dtype=np.float64),
            "F1_macro": np.array(f1m, dtype=np.float64),
            "F1_pos": np.array(f1p, dtype=np.float64),
            "Confusions": np.array(cms, dtype=np.float64),
        }
    return out


def _parse_log_per_seed(log_path: Path) -> Dict[int, Dict[str, Dict[int, float]]]:
    seed_pat = re.compile(r"Running with seed\s+(\d+)")
    train_pat = re.compile(r"TRAIN-(\d+):\s*>>\s*Accuracy:\s*([0-9.]+)\s*F1_score:\s*([0-9.]+)")
    val_pat = re.compile(r"VAL-\([^)]*\)\s*>>\s*Accuracy:\s*([0-9.]+)\s*F1_score:\s*([0-9.]+)")

    cur_seed = None
    cur_epoch = None
    seeds: Dict[int, Dict[str, Dict[int, float]]] = {}

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m_seed = seed_pat.search(line)
        if m_seed:
            cur_seed = int(m_seed.group(1))
            seeds.setdefault(cur_seed, {"train_acc": {}, "train_f1": {}, "val_acc": {}, "val_f1": {}})
            cur_epoch = None
            continue

        if cur_seed is None:
            continue

        m_tr = train_pat.search(line)
        if m_tr:
            ep = int(m_tr.group(1))
            cur_epoch = ep
            seeds[cur_seed]["train_acc"][ep] = float(m_tr.group(2)) * 100.0
            seeds[cur_seed]["train_f1"][ep] = float(m_tr.group(3)) * 100.0
            continue

        m_val = val_pat.search(line)
        if m_val and cur_epoch is not None:
            seeds[cur_seed]["val_acc"][cur_epoch] = float(m_val.group(1)) * 100.0
            seeds[cur_seed]["val_f1"][cur_epoch] = float(m_val.group(2)) * 100.0

    return seeds


def _aggregate_epoch_curves(seed_data: Dict[int, Dict[str, Dict[int, float]]], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_epochs = sorted({ep for sd in seed_data.values() for ep in sd[key].keys()})
    means, stds = [], []
    for ep in all_epochs:
        vals = [sd[key][ep] for sd in seed_data.values() if ep in sd[key]]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return np.array(all_epochs), np.array(means), np.array(stds)


def _plot_learning_curves(camfn_log: Path, muvac_log: Path, out_file: Path, dpi: int):
    cam = _parse_log_per_seed(camfn_log)
    muv = _parse_log_per_seed(muvac_log)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)
    for ax, metric, title in [
        (axes[0], "val_acc", "Validation Accuracy"),
        (axes[1], "val_f1", "Validation F1-score"),
    ]:
        for name, data, color in [("CAMFNet-AML", cam, "#d62728"), ("MuVaC", muv, "#1f77b4")]:
            x, m, s = _aggregate_epoch_curves(data, metric)
            ax.plot(x, m, label=name, color=color, linewidth=2.0)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.20)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=False)

    fig.suptitle("Training Dynamics on MUStARD++ (10 paired seeds)", fontsize=13, fontweight="bold")
    if not getattr(fig, "get_constrained_layout", lambda: False)():
        fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def _plot_seed_distribution(seed_metrics: Dict[str, Dict[str, np.ndarray]], out_file: Path, dpi: int):
    metrics = ["Accuracy", "F1_macro", "F1_pos"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=dpi)
    for i, mt in enumerate(metrics):
        ax = axes[i]
        data = [seed_metrics["camfn"][mt], seed_metrics["muvac"][mt]]
        bp = ax.boxplot(data, tick_labels=["CAMFNet-AML", "MuVaC"], patch_artist=True, widths=0.55)
        bp["boxes"][0].set(facecolor="#f28e8e", alpha=0.65)
        bp["boxes"][1].set(facecolor="#8eb5f2", alpha=0.65)
        for j, arr in enumerate(data, start=1):
            x = np.full(arr.shape[0], j) + np.linspace(-0.08, 0.08, arr.shape[0])
            ax.scatter(x, arr, s=22, alpha=0.8, c="#333333")
        ax.set_title(mt)
        ax.set_ylabel("Score (%)")
        ax.grid(alpha=0.22, linestyle="--")
    fig.suptitle("Seed-wise Performance Distribution", fontsize=13, fontweight="bold")
    if not getattr(fig, "get_constrained_layout", lambda: False)():
        fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_diff(seed_metrics: Dict[str, Dict[str, np.ndarray]], out_file: Path, dpi: int):
    metrics = ["Accuracy", "F1_macro", "F1_pos"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=dpi)
    for i, mt in enumerate(metrics):
        ax = axes[i]
        diff = seed_metrics["camfn"][mt] - seed_metrics["muvac"][mt]
        x = np.arange(1, len(diff) + 1)
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in diff]
        ax.bar(x, diff, color=colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_title(f"Paired Delta: {mt}")
        ax.set_xlabel("Seed Index")
        ax.set_ylabel("CAMFNet-AML - MuVaC (points)")
        ax.grid(alpha=0.22, linestyle="--", axis="y")
        ax.text(0.03, 0.96, f"mean={np.mean(diff):.2f}", transform=ax.transAxes, ha="left", va="top")
    fig.suptitle("Per-seed Paired Improvements", fontsize=13, fontweight="bold")
    if not getattr(fig, "get_constrained_layout", lambda: False)():
        fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def _plot_significance_forest(sig_json: Path, out_file: Path, dpi: int):
    data = json.loads(sig_json.read_text(encoding="utf-8"))
    metrics = ["Accuracy", "F1_macro", "F1_pos"]

    means = [data["tests"][m]["mean_diff_a_minus_b"] for m in metrics]
    ci_lo = [data["tests"][m]["bootstrap"]["ci95_low"] for m in metrics]
    ci_hi = [data["tests"][m]["bootstrap"]["ci95_high"] for m in metrics]
    p_t_holm = [data["tests"][m]["paired_t"]["p_holm"] for m in metrics]

    y = np.arange(len(metrics))[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=dpi)
    for i, yy in enumerate(y):
        m = means[i]
        lo = m - ci_lo[i]
        hi = ci_hi[i] - m
        ax.errorbar(m, yy, xerr=np.array([[lo], [hi]]), fmt="o", color="#d62728", capsize=5, linewidth=2)
        ax.text(ci_hi[i] + 0.12, yy, f"p_holm={p_t_holm[i]:.4f}", va="center", fontsize=10)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(y, metrics)
    ax.set_xlabel("Mean Improvement (CAMFNet-AML - MuVaC, points)")
    ax.set_title("Effect Size with 95% Bootstrap CI")
    ax.grid(alpha=0.22, linestyle="--", axis="x")
    if not getattr(fig, "get_constrained_layout", lambda: False)():
        fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def _plot_holm(sig_json: Path, out_file: Path, dpi: int):
    data = json.loads(sig_json.read_text(encoding="utf-8"))
    metrics = ["Accuracy", "F1_macro", "F1_pos"]
    tests = ["paired_t", "permutation", "wilcoxon"]
    titles = ["Paired t-test", "Permutation", "Wilcoxon"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), dpi=dpi, sharey=True)
    for ax, tname, ttl in zip(axes, tests, titles):
        raw = [data["tests"][m][tname]["p_two_sided"] for m in metrics]
        holm = [data["tests"][m][tname]["p_holm"] for m in metrics]
        x = np.arange(len(metrics))
        w = 0.34
        ax.bar(x - w/2, raw, w, label="raw p", color="#4c78a8")
        ax.bar(x + w/2, holm, w, label="Holm p", color="#f58518")
        ax.axhline(0.05, color="red", linestyle="--", linewidth=1)
        ax.set_xticks(x, metrics, rotation=15)
        ax.set_title(ttl)
        ax.grid(alpha=0.2, linestyle="--", axis="y")
    axes[0].set_ylabel("p-value")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Holm-Bonferroni Correction Across 3 Metrics", fontsize=13, fontweight="bold")
    if not getattr(fig, "get_constrained_layout", lambda: False)():
        fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(seed_metrics: Dict[str, Dict[str, np.ndarray]], out_file: Path, dpi: int):
    cm_cam = seed_metrics["camfn"]["Confusions"].mean(axis=0)
    cm_muv = seed_metrics["muvac"]["Confusions"].mean(axis=0)

    vmax = max(cm_cam.max(), cm_muv.max())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), dpi=dpi, constrained_layout=True)
    for ax, cm, ttl in [(axes[0], cm_cam, "CAMFNet-AML (avg over 10 seeds)"), (axes[1], cm_muv, "MuVaC (avg over 10 seeds)")]:
        im = ax.imshow(cm, cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(ttl)
        ax.set_xticks([0, 1], ["Pred Non-Sarc", "Pred Sarc"], rotation=20, ha="right")
        ax.set_yticks([0, 1], ["True Non-Sarc", "True Sarc"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", color="black", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("Average Count")
    fig.suptitle("Average Confusion Matrices", fontsize=13, fontweight="bold")
    if not getattr(fig, "get_constrained_layout", lambda: False)():
        fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready CAMFNet-AML training/statistics figures.")
    parser.add_argument("--results-csv", type=Path, default=Path("results_sig/normal/mustardpp.csv"))
    parser.add_argument("--sig-json", type=Path, default=Path("results_sig/statistics/mustardpp_camfn_vs_muvac_sig_10seeds.json"))
    parser.add_argument("--camfn-log", type=Path, default=Path("logs_sig/camfn-mustardpp.log"))
    parser.add_argument("--muvac-log", type=Path, default=Path("logs_sig/muvac-mustardpp.log"))
    parser.add_argument("--out-dir", type=Path, default=Path("论文图像/论文统计图"))
    parser.add_argument("--dpi", type=int, default=320)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_metrics = _load_seed_metrics_from_csv(args.results_csv)
    _plot_learning_curves(args.camfn_log, args.muvac_log, args.out_dir / "fig1_learning_curves.png", args.dpi)
    _plot_seed_distribution(seed_metrics, args.out_dir / "fig2_seed_distribution.png", args.dpi)
    _plot_paired_diff(seed_metrics, args.out_dir / "fig3_paired_differences.png", args.dpi)
    _plot_significance_forest(args.sig_json, args.out_dir / "fig4_significance_forest.png", args.dpi)
    _plot_holm(args.sig_json, args.out_dir / "fig5_holm_correction.png", args.dpi)
    _plot_confusion(seed_metrics, args.out_dir / "fig6_confusion_matrices.png", args.dpi)

    manifest = {
        "results_csv": str(args.results_csv),
        "sig_json": str(args.sig_json),
        "camfn_log": str(args.camfn_log),
        "muvac_log": str(args.muvac_log),
        "outputs": [
            "fig1_learning_curves.png",
            "fig2_seed_distribution.png",
            "fig3_paired_differences.png",
            "fig4_significance_forest.png",
            "fig5_holm_correction.png",
            "fig6_confusion_matrices.png",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved all figures to: {args.out_dir}")


if __name__ == "__main__":
    main()

