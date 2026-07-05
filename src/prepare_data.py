"""Build the CEFR benchmark dataset from the public portion of CEFR-SP.

Steps, in order:
1. Download the six official split files (Wiki-Auto and SCoRE subcorpora)
   from the CEFR-SP repository, pinned to an exact commit so upstream
   changes can never silently alter this benchmark.
2. Parse and clean: drop malformed lines, empty sentences, and out-of-range
   labels, then deduplicate within splits and remove any sentence that
   leaks across splits (test is authoritative, then val, then train).
3. Apply the gold-label policy. Each sentence carries labels from two
   expert annotators who never disagree by more than one level in this
   corpus, so disagreements are a tie-break: policy "max" keeps the
   stricter annotator's level (default), "min" keeps the laxer one.
4. Write train/val/test CSVs, a statistics report, and a class
   distribution chart.
5. Verify every raw and processed file against the SHA-256 checksums in
   data_manifest.json, so anyone rerunning the pipeline provably works
   with identical data.

The pipeline is deterministic: no randomness, byte-stable outputs.

Usage:
    .venv/bin/python src/prepare_data.py
    .venv/bin/python src/prepare_data.py --update-manifest   (maintainer only)
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests

from utils import (
    CEFR_LEVELS,
    FIGURES_DIR,
    LABEL_TO_CEFR,
    MANIFEST_PATH,
    METRICS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    load_json,
    save_json,
    sha256_file,
)

PINNED_COMMIT = "2ba4a5005fbff68ab0f863891cf977c50fdcda29"
BASE_URL = (
    "https://raw.githubusercontent.com/yukiar/CEFR-SP/" + PINNED_COMMIT + "/CEFR-SP"
)

SPLITS = ("train", "val", "test")

# (source, split) -> path under BASE_URL. CEFR-SP calls the val split "dev".
RAW_FILES: dict[tuple[str, str], str] = {
    ("score", "train"): "SCoRE/CEFR-SP_SCoRE_train.txt",
    ("score", "val"): "SCoRE/CEFR-SP_SCoRE_dev.txt",
    ("score", "test"): "SCoRE/CEFR-SP_SCoRE_test.txt",
    ("wiki-auto", "train"): "Wiki-Auto/CEFR-SP_Wikiauto_train.txt",
    ("wiki-auto", "val"): "Wiki-Auto/CEFR-SP_Wikiauto_dev.txt",
    ("wiki-auto", "test"): "Wiki-Auto/CEFR-SP_Wikiauto_test.txt",
}

CSV_COLUMNS = ["sentence", "label", "cefr", "source", "label_a", "label_b"]


def download_raw(force: bool = False) -> dict[tuple[str, str], Path]:
    """Download the six official split files into data/raw, reusing cached copies."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[str, str], Path] = {}
    for key, rel in RAW_FILES.items():
        local = RAW_DIR / Path(rel).name
        if force or not local.exists():
            url = f"{BASE_URL}/{rel}"
            print(f"downloading {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            local.write_bytes(resp.content)
        paths[key] = local
    return paths


def parse_raw(
    path: Path, source: str, split: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse one tab-separated raw file, counting every dropped line."""
    dropped = {"malformed": 0, "empty_sentence": 0, "bad_label": 0}
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            dropped["malformed"] += 1
            continue
        sentence = parts[0].strip()
        if not sentence:
            dropped["empty_sentence"] += 1
            continue
        try:
            label_a, label_b = int(parts[1]), int(parts[2])
        except ValueError:
            dropped["bad_label"] += 1
            continue
        if not (1 <= label_a <= 6 and 1 <= label_b <= 6):
            dropped["bad_label"] += 1
            continue
        rows.append(
            {
                "sentence": sentence,
                "label_a": label_a,
                "label_b": label_b,
                "source": source,
            }
        )
    return pd.DataFrame(rows), dropped


def dedupe_key(sentence: str) -> str:
    """Canonical form used only for duplicate detection, never stored."""
    return " ".join(unicodedata.normalize("NFC", sentence).casefold().split())


def dedupe_and_deleak(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Drop within-split duplicates, then drop cross-split overlaps.

    Precedence: test is authoritative, then val, then train. A sentence that
    appears in test is removed from val and train, so no evaluation sentence
    can ever have been seen during training or prompt tuning.
    """
    stats: dict[str, int] = {}
    for split in SPLITS:
        df = frames[split]
        keys = df["sentence"].map(dedupe_key)
        before = len(df)
        frames[split] = df.loc[~keys.duplicated(keep="first")].reset_index(drop=True)
        stats[f"duplicates_within_{split}"] = before - len(frames[split])

    test_keys = set(frames["test"]["sentence"].map(dedupe_key))
    val_keys = frames["val"]["sentence"].map(dedupe_key)
    val_leak = val_keys.isin(test_keys)
    stats["val_overlapping_test"] = int(val_leak.sum())
    frames["val"] = frames["val"].loc[~val_leak.values].reset_index(drop=True)

    protected = test_keys | set(frames["val"]["sentence"].map(dedupe_key))
    train_keys = frames["train"]["sentence"].map(dedupe_key)
    train_leak = train_keys.isin(protected)
    stats["train_overlapping_test_or_val"] = int(train_leak.sum())
    frames["train"] = frames["train"].loc[~train_leak.values].reset_index(drop=True)
    return frames, stats


def agreement_stats(df: pd.DataFrame) -> dict[str, object]:
    """Summarize how often the two expert annotators agree."""
    diff = (df["label_a"] - df["label_b"]).abs()
    pair_counts = (
        df.groupby(["label_a", "label_b"]).size().sort_index()
    )
    return {
        "exact_pct": round(float((diff == 0).mean()) * 100, 2),
        "within_one_level_pct": round(float((diff <= 1).mean()) * 100, 2),
        "max_disagreement_levels": int(diff.max()),
        "pair_counts": {
            f"{a}-{b}": int(n) for (a, b), n in pair_counts.items()
        },
    }


def apply_gold_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Attach the gold label. Policy "max" keeps the stricter annotator's level."""
    df = df.copy()
    pair = df[["label_a", "label_b"]]
    df["label"] = pair.max(axis=1) if policy == "max" else pair.min(axis=1)
    df["cefr"] = df["label"].map(LABEL_TO_CEFR)
    return df


def class_distribution(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    """Count sentences per CEFR level per split."""
    return {
        split: {
            level: int((frames[split]["cefr"] == level).sum())
            for level in CEFR_LEVELS
        }
        for split in SPLITS
    }


def length_stats(df: pd.DataFrame) -> dict[str, float]:
    """Whitespace token length statistics, used later for cost estimates."""
    lengths = df["sentence"].str.split().str.len()
    return {
        "mean": round(float(lengths.mean()), 1),
        "median": float(lengths.median()),
        "p95": float(lengths.quantile(0.95)),
        "max": int(lengths.max()),
    }


def make_figure(dist: dict[str, dict[str, int]], out: Path) -> None:
    """Save a grouped bar chart of class counts per split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.27
    for i, split in enumerate(SPLITS):
        counts = [dist[split][level] for level in CEFR_LEVELS]
        positions = [x + (i - 1) * width for x in range(len(CEFR_LEVELS))]
        bars = ax.bar(positions, counts, width=width, label=split)
        ax.bar_label(bars, fontsize=7, padding=1)
    ax.set_xticks(range(len(CEFR_LEVELS)), CEFR_LEVELS)
    ax.set_ylabel("sentences")
    ax.set_title(
        "CEFR-SP public portion: class distribution per split\n"
        "(gold label: stricter of two expert annotators)"
    )
    ax.legend(title="split", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def build_manifest(
    raw_paths: dict[tuple[str, str], Path], processed_paths: list[Path]
) -> dict[str, object]:
    """Checksums of every file the benchmark depends on."""
    return {
        "source_repo": "https://github.com/yukiar/CEFR-SP",
        "pinned_commit": PINNED_COMMIT,
        "raw": {p.name: sha256_file(p) for p in raw_paths.values()},
        "processed": {p.name: sha256_file(p) for p in processed_paths},
    }


def verify_manifest(actual: dict[str, object]) -> None:
    """Compare freshly computed checksums against the committed manifest."""
    if not MANIFEST_PATH.exists():
        sys.exit(
            "FAIL data_manifest.json not found. If this is intentional "
            "(first generation), rerun with --update-manifest."
        )
    expected = load_json(MANIFEST_PATH)
    if expected == actual:
        print("PASS manifest verified: all files match committed checksums")
        return
    problems: list[str] = []
    for section in ("raw", "processed"):
        exp, act = expected.get(section, {}), actual.get(section, {})
        for name in sorted(set(exp) | set(act)):
            if exp.get(name) != act.get(name):
                problems.append(f"  {section}/{name}")
    sys.exit(
        "FAIL manifest mismatch, the data is not what this benchmark was "
        "built on. Offending files:\n" + "\n".join(problems)
    )


def main() -> None:
    """Run the full pipeline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--policy",
        choices=("max", "min"),
        default="max",
        help="gold label on annotator disagreement: max keeps the stricter "
        "level (default, matches the committed manifest), min the laxer",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="rewrite data_manifest.json from the current outputs",
    )
    parser.add_argument(
        "--force-download", action="store_true", help="redownload raw files"
    )
    args = parser.parse_args()

    raw_paths = download_raw(force=args.force_download)

    dropped_total = {"malformed": 0, "empty_sentence": 0, "bad_label": 0}
    frames: dict[str, pd.DataFrame] = {}
    per_source_counts: dict[str, int] = {}
    for split in SPLITS:
        parts: list[pd.DataFrame] = []
        for source in ("wiki-auto", "score"):
            df, dropped = parse_raw(raw_paths[(source, split)], source, split)
            for key, n in dropped.items():
                dropped_total[key] += n
            per_source_counts[source] = per_source_counts.get(source, 0) + len(df)
            parts.append(df)
        frames[split] = pd.concat(parts, ignore_index=True)
    raw_total = sum(len(f) for f in frames.values())

    frames, dedupe_stats = dedupe_and_deleak(frames)
    agreement = agreement_stats(pd.concat(frames.values(), ignore_index=True))
    frames = {s: apply_gold_policy(f, args.policy) for s, f in frames.items()}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_paths: list[Path] = []
    for split in SPLITS:
        out = PROCESSED_DIR / f"{split}.csv"
        frames[split][CSV_COLUMNS].to_csv(out, index=False, lineterminator="\n")
        processed_paths.append(out)

    dist = class_distribution(frames)
    stats = {
        "source_repo": "https://github.com/yukiar/CEFR-SP",
        "pinned_commit": PINNED_COMMIT,
        "gold_label_policy": args.policy,
        "raw_rows_parsed": raw_total,
        "rows_per_source": per_source_counts,
        "dropped_at_parse": dropped_total,
        "dedupe_and_leakage": dedupe_stats,
        "annotator_agreement": agreement,
        "final_sizes": {s: len(frames[s]) for s in SPLITS},
        "class_distribution": dist,
        "sentence_length_tokens": {
            "all": length_stats(pd.concat(frames.values(), ignore_index=True)),
            "test": length_stats(frames["test"]),
        },
    }
    save_json(METRICS_DIR / "dataset_stats.json", stats)
    make_figure(dist, FIGURES_DIR / "class_distribution.png")

    manifest = build_manifest(raw_paths, processed_paths)
    if args.update_manifest:
        save_json(MANIFEST_PATH, manifest)
        print(f"wrote {MANIFEST_PATH.name}")
    else:
        verify_manifest(manifest)

    print("\n== pipeline summary ==")
    print(f"rows parsed:        {raw_total} (dropped: {dropped_total})")
    print(f"dedupe and leakage: {dedupe_stats}")
    print(
        f"annotator agreement: exact {agreement['exact_pct']}%, "
        f"within one level {agreement['within_one_level_pct']}%"
    )
    print(f"gold label policy:  {args.policy}")
    print(f"final sizes:        {stats['final_sizes']}")
    header = "  ".join(f"{lvl:>5}" for lvl in CEFR_LEVELS)
    print(f"\nclass distribution   {header}")
    for split in SPLITS:
        row = "  ".join(f"{dist[split][lvl]:>5}" for lvl in CEFR_LEVELS)
        print(f"{split:>18}   {row}")
    print(
        "\noutputs: data/processed/{train,val,test}.csv, "
        "results/metrics/dataset_stats.json, "
        "results/figures/class_distribution.png"
    )


if __name__ == "__main__":
    main()
