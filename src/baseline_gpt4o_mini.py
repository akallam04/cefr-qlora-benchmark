"""GPT-4o-mini few-shot baseline for CEFR sentence classification.

Builds a fixed few-shot prompt from training exemplars (never val or test),
classifies one split of the benchmark, and writes a prediction file in the
shared format the evaluation harness consumes. Latency is measured per
request and cost is computed from the exact token counts the API returns,
never from estimates.

Methodology discipline: the prompt is engineered against the val split
only. Once frozen, the test split is classified exactly once.

Failures are honest: a response that cannot be parsed into one of the six
labels, or a request that exhausts retries, is recorded with pred null and
counts as wrong in every metric.

Usage:
    .venv/bin/python src/baseline_gpt4o_mini.py --dry-run
    .venv/bin/python src/baseline_gpt4o_mini.py --split val --limit 150
    .venv/bin/python src/baseline_gpt4o_mini.py --split test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from utils import (
    CEFR_LEVELS,
    DATA_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    REPO_ROOT,
    SEED,
    save_json,
    sha256_file,
)

MODEL_DEFAULT = "gpt-4o-mini"
PROMPT_VERSION = "v2"
SHOTS_PER_CLASS = 3
MAX_ATTEMPTS = 7
REQUEST_TIMEOUT_S = 30.0

# gpt-4o-mini prices as of 2026-07-05. raw token counts go into the
# output json so cost can be recomputed if prices move
PRICING_USD_PER_1M = {"input": 0.15, "output": 0.60, "as_of": "2026-07-05"}

LABEL_RE = re.compile(r"\b(A1|A2|B1|B2|C1|C2)\b")

SYSTEM_PROMPT = """\
You are an expert CEFR examiner for English as a foreign language.
Rate the difficulty of one English sentence at a time: the CEFR level a
learner needs in order to fully understand the sentence.

Level guide:
A1: very short, concrete, everyday statements; basic vocabulary; simple present or simple past.
A2: everyday questions and statements about activities, plans, and preferences; may join two clauses with and, but, because, or a simple when/if clause.
B1: longer sentences on familiar topics; subordinate clauses; everyday idioms; past perfect or common phrasal verbs.
B2: complex sentences with multiple clauses; abstract topics; broad vocabulary.
C1: dense sentences with multiple subordinate clauses or heavy noun phrases; technical, scientific, or low-frequency vocabulary; formal or specialized structures.
C2: highly complex or subtle sentences; rare vocabulary; near-native constructions.

Calibration:
- These labels follow a strict examiner standard: a sentence is rated at the level a learner needs to understand it fully, including its hardest word or structure.
- One low-frequency word or one complex structure is enough to lift an otherwise simple sentence to the higher level.
- If you hesitate between two adjacent levels, choose the higher one, unless the sentence clearly contains nothing beyond the lower level.

Rules:
- Sentences may contain tokenization artifacts such as spaces before punctuation. Ignore them.
- Judge the language of the sentence itself, not how difficult its topic is.
- Answer with exactly one of: A1, A2, B1, B2, C1, C2. Output nothing else."""


def select_exemplars(train: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """Pick prototype sentences per level for the few-shot prompt.

    Preference order: sentences both experts agreed on (unambiguous
    prototypes), cleanly formed (no truncated starts), medium length
    (5 to 30 tokens) for prompt economy, and drawn from both subcorpora
    so no level is represented by a single sentence template. Seeded
    sampling on a deterministic frame, so the prompt is identical on
    every run.
    """
    chosen: list[pd.DataFrame] = []
    for level in range(1, 7):
        pool = train[(train["label"] == level) & (train["label_a"] == train["label_b"])]
        if len(pool) < per_class:
            pool = train[train["label"] == level]
        lengths = pool["sentence"].str.split().str.len()
        medium = pool[(lengths >= 5) & (lengths <= 30)]
        if len(medium) >= per_class:
            pool = medium
        clean = pool[pool["sentence"].str.match(r"[A-Z0-9]")]
        if len(clean) >= per_class:
            pool = clean

        parts: list[pd.DataFrame] = []
        sources = sorted(pool["source"].unique())
        for j, source in enumerate(sources):
            want = per_class // len(sources) + (1 if j < per_class % len(sources) else 0)
            sub = pool[pool["source"] == source]
            parts.append(sub.sample(n=min(want, len(sub)), random_state=SEED))
        picked = pd.concat(parts)
        if len(picked) < per_class:
            rest = pool.drop(picked.index).sample(
                n=per_class - len(picked), random_state=SEED
            )
            picked = pd.concat([picked, rest])
        chosen.append(picked)
    return pd.concat(chosen)


def build_prefix(exemplars: pd.DataFrame) -> list[dict[str, str]]:
    """System prompt plus few-shot pairs, ordered A1 to C2."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for _, row in exemplars.sort_values("label", kind="stable").iterrows():
        messages.append({"role": "user", "content": row["sentence"]})
        messages.append({"role": "assistant", "content": row["cefr"]})
    return messages


def classify_one(
    client: "object", prefix: list[dict[str, str]], index: int, sentence: str, model: str
) -> dict[str, object]:
    """Classify one sentence with retries. Latency covers the successful attempt only."""
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    retryable = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    messages = prefix + [{"role": "user", "content": sentence}]
    retries = 0
    for attempt in range(MAX_ATTEMPTS):
        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=5,
                seed=SEED,
                timeout=REQUEST_TIMEOUT_S,
            )
        except retryable as exc:
            retries += 1
            if attempt == MAX_ATTEMPTS - 1:
                return {
                    "index": index,
                    "pred": None,
                    "raw": f"api_error: {type(exc).__name__}",
                    "latency_s": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "retries": retries,
                    "api_model": None,
                }
            time.sleep(min(2**attempt + random.random(), 30.0))
            continue
        latency = time.perf_counter() - start
        raw = (resp.choices[0].message.content or "").strip()
        match = LABEL_RE.search(raw.upper())
        return {
            "index": index,
            "pred": match.group(1) if match else None,
            "raw": raw[:40],
            "latency_s": round(latency, 4),
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "retries": retries,
            "api_model": resp.model,
        }
    raise RuntimeError("unreachable")


def quick_metrics(golds: list[str], preds: list[str | None]) -> dict[str, float]:
    """Sanity metrics printed after a run. The eval harness is authoritative."""
    from sklearn.metrics import f1_score

    n = len(golds)
    exact = sum(g == p for g, p in zip(golds, preds))
    within = 0
    order = {lvl: i for i, lvl in enumerate(CEFR_LEVELS)}
    for g, p in zip(golds, preds):
        if p is not None and abs(order[g] - order[p]) <= 1:
            within += 1
    mapped = [p if p is not None else "FAIL" for p in preds]
    macro = f1_score(golds, mapped, labels=CEFR_LEVELS, average="macro", zero_division=0)
    return {
        "accuracy_pct": round(100 * exact / n, 2),
        "within_one_level_pct": round(100 * within / n, 2),
        "macro_f1": round(float(macro), 4),
    }


def dry_run(prefix: list[dict[str, str]], df: pd.DataFrame, n_eval: int) -> None:
    """Print the frozen prompt and a cost estimate without calling the API."""
    print("== system prompt ==")
    print(prefix[0]["content"])
    print("\n== few-shot exemplars ==")
    for user, assistant in zip(prefix[1::2], prefix[2::2]):
        print(f"  [{assistant['content']}] {user['content']}")
    example = df.iloc[0]["sentence"]
    print(f"\n== example final user message ==\n  {example}")
    prefix_chars = sum(len(m["content"]) for m in prefix)
    mean_sentence_chars = float(df["sentence"].str.len().mean())
    est_prompt_tokens = (prefix_chars + mean_sentence_chars) / 4
    est_cost = n_eval * est_prompt_tokens * PRICING_USD_PER_1M["input"] / 1e6
    print(f"\nestimated prompt tokens per request: ~{est_prompt_tokens:.0f}")
    print(f"estimated input cost for {n_eval} requests: ~${est_cost:.2f} plus output tokens")


def main() -> None:
    """Run the baseline over one split, with resume support."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=None, help="classify only the first N rows")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--dry-run", action="store_true", help="print the prompt and cost estimate, no API calls")
    args = parser.parse_args()

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    split_path = PROCESSED_DIR / f"{args.split}.csv"
    df = pd.read_csv(split_path)
    if args.limit and args.limit < len(df):
        # csvs are ordered by class, a head slice would be almost all B1:
        # seeded stratified sample instead, original row indices kept
        parts = [
            group.sample(
                n=min(len(group), max(1, round(len(group) * args.limit / len(df)))),
                random_state=SEED,
            )
            for _, group in df.groupby("cefr")
        ]
        df = pd.concat(parts).sort_index()

    exemplars = select_exemplars(train, SHOTS_PER_CLASS)
    prefix = build_prefix(exemplars)

    if args.dry_run:
        dry_run(prefix, df, len(df))
        return

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        sys.exit("OPENAI_API_KEY missing from .env, run scripts/preflight.py first")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, max_retries=0)

    interim_path = (
        DATA_DIR / "interim" / f"baseline_{args.model}_{args.split}_{PROMPT_VERSION}.jsonl"
    )
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict[str, object]] = {}
    if interim_path.exists():
        for line in interim_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["index"]] = rec
        print(f"resuming: {len(done)} of {len(df)} rows already classified")

    todo = [(int(i), row["sentence"]) for i, row in df.iterrows() if int(i) not in done]
    lock = threading.Lock()
    if todo:
        with interim_path.open("a", encoding="utf-8") as sink:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(classify_one, client, prefix, i, s, args.model)
                    for i, s in todo
                ]
                for future in tqdm(as_completed(futures), total=len(futures), unit="req"):
                    rec = future.result()
                    with lock:
                        sink.write(json.dumps(rec) + "\n")
                        sink.flush()
                    done[rec["index"]] = rec

    records = [done[int(i)] for i in df.index]
    golds = df["cefr"].tolist()
    preds = [r["pred"] for r in records]
    for rec, gold in zip(records, golds):
        rec["gold"] = gold

    latencies = [r["latency_s"] for r in records if r["latency_s"] is not None]
    in_tokens = int(sum(r["prompt_tokens"] for r in records))
    out_tokens = int(sum(r["completion_tokens"] for r in records))
    cost = (
        in_tokens * PRICING_USD_PER_1M["input"] + out_tokens * PRICING_USD_PER_1M["output"]
    ) / 1e6
    api_models = {r["api_model"] for r in records if r["api_model"]}
    metrics = quick_metrics(golds, preds)
    aggregate = {
        **metrics,
        "n": len(records),
        "parse_failures": sum(1 for r in records if r["pred"] is None and r["latency_s"] is not None),
        "api_failures": sum(1 for r in records if r["latency_s"] is None),
        "total_retries": int(sum(r["retries"] for r in records)),
        "total_prompt_tokens": in_tokens,
        "total_completion_tokens": out_tokens,
        "latency_s": {
            "mean": round(float(np.mean(latencies)), 3),
            "median": round(float(np.median(latencies)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
        },
        "total_cost_usd": round(cost, 4),
        "cost_per_1k_requests_usd": round(cost / len(records) * 1000, 4),
    }

    payload = {
        "model": args.model,
        "api_model_versions": sorted(api_models),
        "split": args.split,
        "prompt_version": PROMPT_VERSION,
        "config": {
            "temperature": 0.0,
            "max_completion_tokens": 5,
            "seed": SEED,
            "shots_per_class": SHOTS_PER_CLASS,
            "exemplar_train_rows": [int(i) for i in exemplars.index],
            "workers": args.workers,
            "limit": args.limit,
        },
        "pricing_usd_per_1m": PRICING_USD_PER_1M,
        "dataset": {"split_csv_sha256": sha256_file(split_path), "n": len(records)},
        "aggregate": aggregate,
        "predictions": [
            {k: r[k] for k in ("index", "gold", "pred", "latency_s", "prompt_tokens", "completion_tokens", "retries")}
            for r in sorted(records, key=lambda r: r["index"])
        ],
    }
    out_path = PREDICTIONS_DIR / f"gpt4o_mini_{args.split}.json"
    save_json(out_path, payload)

    print(f"\n== baseline summary ({args.model}, split={args.split}, n={len(records)}) ==")
    for key in ("accuracy_pct", "within_one_level_pct", "macro_f1"):
        print(f"{key}: {aggregate[key]}")
    print(f"parse failures: {aggregate['parse_failures']}, api failures: {aggregate['api_failures']}, retries: {aggregate['total_retries']}")
    print(f"latency s: {aggregate['latency_s']}")
    print(f"tokens: {in_tokens} in, {out_tokens} out")
    print(f"cost: ${aggregate['total_cost_usd']} total, ${aggregate['cost_per_1k_requests_usd']} per 1k requests")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
