"""Post-training pipeline: pick the best checkpoint, predict, and publish.

Runs after finetune_qlora.py or the Colab notebook has produced epoch
checkpoints. Loads the 4-bit base once, hot-swaps each checkpoint adapter,
scores them all on val by macro F1, then uses the winner to write the val
and test prediction files in the shared format. With --upload, pushes the
winning adapter plus both prediction files to the Hub so no browser
download is involved.

Usage (CUDA GPU required):
    python src/select_and_predict.py --checkpoints-dir /content/qlora-out --upload
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from predict_finetuned import MODEL_ID, label_token_ids, predict_levels
from utils import (
    LABEL_TO_CEFR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    SEED,
    save_json,
    sha256_file,
)


def build_payload(
    df: pd.DataFrame,
    preds: list[int],
    split: str,
    wall_s: float,
    latencies: list[float],
    best_name: str,
    scores: dict[str, float],
    gpu: str,
    adapter_repo: str,
    batch_size: int,
) -> dict:
    """Shared prediction format, same schema as the baseline files."""
    from sklearn.metrics import accuracy_score, f1_score

    golds = df["label"].tolist()
    within = float(np.mean([abs(g - p) <= 1 for g, p in zip(golds, preds)]))
    return {
        "model": "llama-3-8b-qlora",
        "adapter_repo": adapter_repo,
        "selected_checkpoint": best_name,
        "gpu": gpu,
        "split": split,
        "config": {
            "model_id": MODEL_ID,
            "batch_size": batch_size,
            "checkpoint_val_macro_f1": {k: round(v, 4) for k, v in scores.items()},
        },
        "dataset": {
            "split_csv_sha256": sha256_file(PROCESSED_DIR / f"{split}.csv"),
            "n": len(df),
        },
        "aggregate": {
            "accuracy_pct": round(100 * accuracy_score(golds, preds), 2),
            "within_one_level_pct": round(100 * within, 2),
            "macro_f1": round(float(f1_score(golds, preds, average="macro")), 4),
            "parse_failures": 0,
            "batch_size": batch_size,
            "batched_throughput_sentences_per_s": round(len(df) / wall_s, 2),
            "latency_s": {
                "mean": round(float(np.mean(latencies)), 3),
                "median": round(float(np.median(latencies)), 3),
                "p95": round(float(np.percentile(latencies, 95)), 3),
            },
        },
        "predictions": [
            {"index": int(i), "gold": LABEL_TO_CEFR[g], "pred": LABEL_TO_CEFR[p]}
            for i, g, p in zip(df.index, golds, preds)
        ],
    }


def main() -> None:
    """Select the best checkpoint on val, predict both splits, publish."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--adapter-repo", default="akallam04/Llama-3-8B-cefr-qlora")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-sample", type=int, default=100)
    parser.add_argument("--upload", action="store_true", help="push adapter and prediction files to the Hub")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required.")
    from peft import PeftModel
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.getenv("HF_TOKEN", "").strip() or None
    checkpoints = sorted(
        glob.glob(f"{args.checkpoints_dir}/checkpoint-*"),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        sys.exit(f"no checkpoints under {args.checkpoints_dir}")
    print(f"found {len(checkpoints)} checkpoints")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = label_token_ids(tokenizer)

    # capability >= 8 (ampere+) for native bf16, T4 only emulates it
    compute_dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        dtype=compute_dtype,
        device_map={"": 0},
        token=token,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    gpu = torch.cuda.get_device_name(0)

    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    names = [c.rsplit("/", 1)[-1] for c in checkpoints]
    model = PeftModel.from_pretrained(base, checkpoints[0], adapter_name=names[0])
    for ckpt, name in zip(checkpoints[1:], names[1:]):
        model.load_adapter(ckpt, adapter_name=name)
    model.eval()

    scores: dict[str, float] = {}
    for ckpt, name in zip(checkpoints, names):
        model.set_adapter(name)
        preds = predict_levels(model, tokenizer, label_ids, val_df["sentence"].tolist(), args.batch_size)
        acc = accuracy_score(val_df["label"], preds)
        macro = float(f1_score(val_df["label"], preds, average="macro"))
        scores[name] = macro
        print(f"{name}: val accuracy {acc:.4f}, val macro F1 {macro:.4f}")

    best = max(scores, key=scores.get)
    best_dir = f"{args.checkpoints_dir}/{best}"
    model.set_adapter(best)
    print(f"selected {best} (val macro F1 {scores[best]:.4f})")

    for split, df in (("val", val_df), ("test", test_df)):
        sentences = df["sentence"].tolist()
        start = time.perf_counter()
        preds = predict_levels(model, tokenizer, label_ids, sentences, args.batch_size)
        wall = time.perf_counter() - start
        sample = df.sample(n=min(args.latency_sample, len(df)), random_state=SEED)
        latencies: list[float] = []
        for sentence in sample["sentence"]:
            t0 = time.perf_counter()
            predict_levels(model, tokenizer, label_ids, [sentence], 1)
            latencies.append(time.perf_counter() - t0)
        payload = build_payload(
            df, preds, split, wall, latencies, best, scores, gpu,
            args.adapter_repo, args.batch_size,
        )
        out = PREDICTIONS_DIR / f"llama3_qlora_{split}.json"
        save_json(out, payload)
        agg = payload["aggregate"]
        print(
            f"{split}: accuracy {agg['accuracy_pct']}%, macro F1 {agg['macro_f1']}, "
            f"throughput {agg['batched_throughput_sentences_per_s']}/s, wrote {out}"
        )

    if args.upload:
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)
        create_repo(args.adapter_repo, exist_ok=True, token=token)
        api.upload_folder(
            folder_path=best_dir,
            repo_id=args.adapter_repo,
            allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
        )
        for split in ("val", "test"):
            api.upload_file(
                path_or_fileobj=str(PREDICTIONS_DIR / f"llama3_qlora_{split}.json"),
                path_in_repo=f"benchmark/llama3_qlora_{split}.json",
                repo_id=args.adapter_repo,
            )
        print(f"uploaded adapter and prediction files to {args.adapter_repo}")


if __name__ == "__main__":
    main()
