"""Compare two models' predictions on the identical benchmark split.

Loads two prediction files in the shared format, proves both were produced
on byte-identical data (split CSV SHA-256 plus row-by-row gold check),
recomputes every metric from the raw predictions, runs an exact McNemar
test on paired correctness, and writes a markdown table, charts, and a
metrics json.

Usage:
    .venv/bin/python src/evaluate.py
    .venv/bin/python src/evaluate.py --split val --gpu-usd-per-hour 0.45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from utils import (
    CEFR_LEVELS,
    CEFR_TO_LABEL,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    load_json,
    save_json,
)

# representative on-demand $/hr, cloud.google.com GPU pricing, checked 2026-07-05
GPU_USD_PER_HOUR = {"T4": 0.35, "L4": 0.70, "A100": 3.67}
LABELS = list(range(1, 7))


def load_predictions(path: Path) -> dict:
    """Read one prediction file and attach aligned gold/pred label arrays."""
    payload = load_json(path)
    rows = sorted(payload["predictions"], key=lambda r: r["index"])
    payload["_indices"] = [r["index"] for r in rows]
    payload["_golds"] = [CEFR_TO_LABEL[r["gold"]] for r in rows]
    # 0 marks a parse or api failure, counted as wrong everywhere
    payload["_preds"] = [CEFR_TO_LABEL.get(r["pred"], 0) for r in rows]
    return payload


def check_identical_dataset(a: dict, b: dict) -> None:
    """Both files must carry the same split checksum, rows, and golds."""
    sha_a = a["dataset"]["split_csv_sha256"]
    sha_b = b["dataset"]["split_csv_sha256"]
    if sha_a != sha_b:
        sys.exit(f"FAIL dataset mismatch: {sha_a[:12]} vs {sha_b[:12]}")
    if a["_indices"] != b["_indices"] or a["_golds"] != b["_golds"]:
        sys.exit("FAIL prediction files cover different rows or gold labels")
    print(
        f"PASS both models evaluated on identical data "
        f"(sha {sha_a[:12]}, n={len(a['_golds'])})"
    )


def core_metrics(golds: list[int], preds: list[int]) -> dict:
    """Everything recomputed from raw predictions, failures count as wrong."""
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

    g, p = np.array(golds), np.array(preds)
    valid = p > 0
    per_class = f1_score(g, p, labels=LABELS, average=None, zero_division=0)
    qwk = (
        cohen_kappa_score(g[valid], p[valid], weights="quadratic")
        if valid.any()
        else 0.0
    )
    return {
        "accuracy_pct": round(100 * float((g == p).mean()), 2),
        "within_one_level_pct": round(100 * float(((np.abs(g - p) <= 1) & valid).mean()), 2),
        "macro_f1": round(float(f1_score(g, p, labels=LABELS, average="macro", zero_division=0)), 4),
        "per_class_f1": {lvl: round(float(s), 4) for lvl, s in zip(CEFR_LEVELS, per_class)},
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "mae_levels": round(float(np.abs(g[valid] - p[valid]).mean()), 4) if valid.any() else None,
        "failures": int((~valid).sum()),
        "confusion_matrix": confusion_matrix(g, p, labels=LABELS).tolist(),
    }


def mcnemar_exact(golds: list[int], preds_a: list[int], preds_b: list[int]) -> dict:
    """Exact McNemar on discordant pairs: is the accuracy gap real."""
    from scipy.stats import binomtest

    g = np.array(golds)
    a_right = np.array(preds_a) == g
    b_right = np.array(preds_b) == g
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    n = only_a + only_b
    p_value = binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
    return {
        "only_baseline_correct": only_a,
        "only_finetuned_correct": only_b,
        "p_value": float(p_value),
    }


def cost_block(payload: dict, gpu_price_override: float | None) -> dict:
    """Cost per 1k requests: API models from measured tokens, local models
    from measured throughput times a GPU rental price."""
    agg = payload["aggregate"]
    if "cost_per_1k_requests_usd" in agg:
        return {
            "usd_per_1k_requests": agg["cost_per_1k_requests_usd"],
            "basis": "measured tokens x published API price",
        }
    throughput = agg["batched_throughput_sentences_per_s"]
    gpu = payload.get("gpu", "")
    price = gpu_price_override or next(
        (v for k, v in GPU_USD_PER_HOUR.items() if k in gpu), None
    )
    if price is None:
        sys.exit(f"no $/hr known for gpu {gpu!r}, pass --gpu-usd-per-hour")
    return {
        "usd_per_1k_requests": round(price / 3600.0 / throughput * 1000.0, 4),
        "basis": f"{gpu} at ${price}/h on-demand, {throughput} sentences/s batched, full utilization",
        "gpu_usd_per_hour": price,
    }


def make_figures(a: dict, b: dict, ma: dict, mb: dict, ca: dict, cb: dict, figures_dir: Path) -> None:
    """Four comparison charts for the README."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    name_a, name_b = a["model"], b["model"]

    fig, ax = plt.subplots(figsize=(7, 4))
    keys = ["accuracy_pct", "within_one_level_pct"]
    ticks = ["exact accuracy %", "within one level %", "macro F1 x100"]
    vals_a = [ma[k] for k in keys] + [100 * ma["macro_f1"]]
    vals_b = [mb[k] for k in keys] + [100 * mb["macro_f1"]]
    x = np.arange(3)
    r1 = ax.bar(x - 0.18, vals_a, width=0.36, label=name_a)
    r2 = ax.bar(x + 0.18, vals_b, width=0.36, label=name_b)
    ax.bar_label(r1, fmt="%.1f", fontsize=8)
    ax.bar_label(r2, fmt="%.1f", fontsize=8)
    ax.set_xticks(x, ticks)
    ax.set_title("Headline metrics on the identical test set")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "metrics_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, metrics, name in ((axes[0], ma, name_a), (axes[1], mb, name_b)):
        cm = np.array(metrics["confusion_matrix"])
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(6), CEFR_LEVELS)
        ax.set_yticks(range(6), CEFR_LEVELS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
        ax.set_title(name)
    fig.suptitle("Confusion matrices (counts, row-normalized color)")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(6)
    r1 = ax.bar(x - 0.18, list(ma["per_class_f1"].values()), width=0.36, label=name_a)
    r2 = ax.bar(x + 0.18, list(mb["per_class_f1"].values()), width=0.36, label=name_b)
    ax.bar_label(r1, fmt="%.2f", fontsize=7)
    ax.bar_label(r2, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, CEFR_LEVELS)
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1: the extremes decide macro F1")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "per_class_f1.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    lat_a, lat_b = a["aggregate"]["latency_s"], b["aggregate"]["latency_s"]
    x = np.arange(2)
    axes[0].bar(x - 0.18, [lat_a["mean"], lat_a["p95"]], width=0.36, label=name_a)
    axes[0].bar(x + 0.18, [lat_b["mean"], lat_b["p95"]], width=0.36, label=name_b)
    axes[0].set_xticks(x, ["mean", "p95"])
    axes[0].set_ylabel("seconds per request")
    axes[0].set_title("Single-request latency")
    axes[0].legend(frameon=False)
    bars = axes[1].bar(
        [name_a, name_b],
        [ca["usd_per_1k_requests"], cb["usd_per_1k_requests"]],
        color=["tab:blue", "tab:orange"],
        width=0.5,
    )
    axes[1].bar_label(bars, fmt="$%.3f", fontsize=9)
    axes[1].set_title("Cost per 1,000 requests")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "latency_cost.png", dpi=150)
    plt.close(fig)


def markdown_table(a: dict, b: dict, ma: dict, mb: dict, ca: dict, cb: dict) -> str:
    """README-ready comparison table."""
    lat_a, lat_b = a["aggregate"]["latency_s"], b["aggregate"]["latency_s"]
    rows = [
        ("Exact accuracy", f"{ma['accuracy_pct']}%", f"{mb['accuracy_pct']}%"),
        ("Within one level", f"{ma['within_one_level_pct']}%", f"{mb['within_one_level_pct']}%"),
        ("Macro F1", f"{ma['macro_f1']}", f"{mb['macro_f1']}"),
        ("Quadratic weighted kappa", f"{ma['quadratic_weighted_kappa']}", f"{mb['quadratic_weighted_kappa']}"),
        ("MAE (levels)", f"{ma['mae_levels']}", f"{mb['mae_levels']}"),
        ("Latency mean / p95 (s)", f"{lat_a['mean']} / {lat_a['p95']}", f"{lat_b['mean']} / {lat_b['p95']}"),
        ("Cost per 1k requests", f"${ca['usd_per_1k_requests']}", f"${cb['usd_per_1k_requests']}"),
        ("Failures", str(ma["failures"]), str(mb["failures"])),
    ]
    lines = [f"| Metric | {a['model']} | {b['model']} |", "|---|---|---|"]
    lines += [f"| {name} | {va} | {vb} |" for name, va, vb in rows]
    return "\n".join(lines)


def main() -> None:
    """Run the comparison and write table, charts, and metrics json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--finetuned", type=Path, default=None)
    parser.add_argument("--gpu-usd-per-hour", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="override output root (smoke tests)")
    args = parser.parse_args()

    baseline_path = args.baseline or PREDICTIONS_DIR / f"gpt4o_mini_{args.split}.json"
    finetuned_path = args.finetuned or PREDICTIONS_DIR / f"llama3_qlora_{args.split}.json"
    metrics_dir = (args.out_dir / "metrics") if args.out_dir else METRICS_DIR
    figures_dir = (args.out_dir / "figures") if args.out_dir else FIGURES_DIR

    a = load_predictions(baseline_path)
    b = load_predictions(finetuned_path)
    check_identical_dataset(a, b)

    golds = a["_golds"]
    ma = core_metrics(golds, a["_preds"])
    mb = core_metrics(golds, b["_preds"])
    ca = cost_block(a, args.gpu_usd_per_hour)
    cb = cost_block(b, args.gpu_usd_per_hour)
    mc = mcnemar_exact(golds, a["_preds"], b["_preds"])

    make_figures(a, b, ma, mb, ca, cb, figures_dir)
    comparison = {
        "split": args.split,
        "n": len(golds),
        "dataset_sha256": a["dataset"]["split_csv_sha256"],
        "models": {
            a["model"]: {"metrics": ma, "cost": ca, "latency_s": a["aggregate"]["latency_s"],
                          "versions": a.get("api_model_versions") or a.get("selected_checkpoint")},
            b["model"]: {"metrics": mb, "cost": cb, "latency_s": b["aggregate"]["latency_s"],
                          "versions": b.get("api_model_versions") or b.get("selected_checkpoint"),
                          "gpu": b.get("gpu")},
        },
        "mcnemar": mc,
    }
    save_json(metrics_dir / f"comparison_{args.split}.json", comparison)

    print()
    print(markdown_table(a, b, ma, mb, ca, cb))
    print()
    print(
        f"McNemar exact test: {mc['only_finetuned_correct']} sentences only the "
        f"fine-tune got right vs {mc['only_baseline_correct']} only the baseline, "
        f"p = {mc['p_value']:.2e}"
    )
    print(f"\nwrote {metrics_dir / f'comparison_{args.split}.json'} and 4 figures in {figures_dir}")


if __name__ == "__main__":
    main()
