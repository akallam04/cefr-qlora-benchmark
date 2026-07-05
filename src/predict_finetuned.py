"""Predict CEFR levels with the QLoRA adapter on a CUDA machine.

Script twin of the prediction cells in notebooks/finetune_qlora_colab.ipynb.
Loads the 4-bit base model plus a LoRA adapter (local checkpoint dir or Hub
repo), classifies one split with a single forward pass per batch (argmax
over the six label-token logits, no generation, no parsing), measures
batched throughput and single-stream latency, and writes the shared
prediction format consumed by src/evaluate.py.

Usage (CUDA GPU required):
    python src/predict_finetuned.py --adapter akallam04/Llama-3-8B-cefr-qlora --split test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    LABEL_TO_CEFR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    SEED,
    format_prompt,
    save_json,
    sha256_file,
)

MODEL_ID = "meta-llama/Meta-Llama-3-8B"


def label_token_ids(tokenizer) -> list[int]:
    """Token ids of the six single-token labels ' 1' to ' 6'."""
    ids: list[int] = []
    for digit in "123456":
        enc = tokenizer.encode(" " + digit, add_special_tokens=False)
        assert len(enc) == 1, f"label ' {digit}' is not a single token: {enc}"
        ids.append(enc[0])
    return ids


def load_model_and_tokenizer(adapter: str, token: str | None):
    """Load the 4-bit NF4 base model with the adapter applied, eval mode."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        dtype=compute_dtype,
        device_map={"": 0},
        token=token,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, adapter, token=token)
    model.eval()
    return model, tokenizer


def predict_levels(
    model, tokenizer, label_ids: list[int], sentences: list[str], batch_size: int
) -> list[int]:
    """Classify sentences: one forward pass, argmax over six label logits."""
    import torch

    tokenizer.padding_side = "left"
    preds: list[int] = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            prompts = [format_prompt(s) for s in sentences[start : start + batch_size]]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            logits = model(**enc).logits[:, -1, :]
            preds.extend((logits[:, label_ids].argmax(dim=-1) + 1).tolist())
    return preds


def main() -> None:
    """Classify one split and write the shared prediction file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default="akallam04/Llama-3-8B-cefr-qlora")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-sample", type=int, default=100)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required. Use the Colab notebook or any CUDA machine.")
    from sklearn.metrics import accuracy_score, f1_score

    token = os.getenv("HF_TOKEN", "").strip() or None
    csv_path = PROCESSED_DIR / f"{args.split}.csv"
    df = pd.read_csv(csv_path)
    model, tokenizer = load_model_and_tokenizer(args.adapter, token)
    label_ids = label_token_ids(tokenizer)

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

    golds = df["label"].tolist()
    within = float(np.mean([abs(g - p) <= 1 for g, p in zip(golds, preds)]))
    payload = {
        "model": "llama-3-8b-qlora",
        "adapter_repo": args.adapter,
        "gpu": torch.cuda.get_device_name(0),
        "split": args.split,
        "config": {"model_id": MODEL_ID, "batch_size": args.batch_size},
        "dataset": {"split_csv_sha256": sha256_file(csv_path), "n": len(df)},
        "aggregate": {
            "accuracy_pct": round(100 * accuracy_score(golds, preds), 2),
            "within_one_level_pct": round(100 * within, 2),
            "macro_f1": round(float(f1_score(golds, preds, average="macro")), 4),
            "parse_failures": 0,
            "batch_size": args.batch_size,
            "batched_throughput_sentences_per_s": round(len(df) / wall, 2),
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
    out_path = PREDICTIONS_DIR / f"llama3_qlora_{args.split}.json"
    save_json(out_path, payload)
    agg = payload["aggregate"]
    print(
        f"{args.split}: accuracy {agg['accuracy_pct']}%, macro F1 {agg['macro_f1']}, "
        f"throughput {agg['batched_throughput_sentences_per_s']}/s, wrote {out_path}"
    )


if __name__ == "__main__":
    main()
