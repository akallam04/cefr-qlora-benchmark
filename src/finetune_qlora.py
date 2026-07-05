"""QLoRA fine-tune of Llama 3 8B for CEFR classification (script twin).

Equivalent to notebooks/finetune_qlora_colab.ipynb for any CUDA machine:
4-bit NF4 base, LoRA rank 16 on all attention and MLP projections,
completion-only loss on a single label token, epoch checkpoints scored on
val by macro F1. The notebook is the canonical runnable artifact; this
script exists for GPU servers. Prediction and export live in
src/predict_finetuned.py.

Usage (CUDA GPU required):
    HF_TOKEN=... WANDB_API_KEY=... python src/finetune_qlora.py --output-dir qlora-out
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

from predict_finetuned import label_token_ids, predict_levels
from utils import PROCESSED_DIR, SEED, completion_for, format_prompt

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    """Train the adapter and report the best epoch checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="qlora-out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required. Use the Colab notebook or any CUDA machine.")
    from peft import LoraConfig
    from sklearn.metrics import accuracy_score, f1_score
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    token = os.getenv("HF_TOKEN", "").strip() or None
    # capability >= 8 (ampere+) for native bf16, T4 only emulates it
    bf16_ok = torch.cuda.get_device_capability(0)[0] >= 8
    per_device = 8 if bf16_ok else 4
    grad_accum = max(1, args.effective_batch // per_device)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")

    def to_example(row: pd.Series) -> dict[str, str]:
        return {
            "prompt": format_prompt(row["sentence"]),
            "completion": completion_for(row["label"]),
        }

    train_ds = Dataset.from_list(
        [to_example(r) for _, r in train_df.iterrows()]
    ).shuffle(seed=SEED)
    val_ds = Dataset.from_list([to_example(r) for _, r in val_df.iterrows()])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = label_token_ids(tokenizer)
    print("label token ids:", label_ids)

    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        dtype=compute_dtype,
        device_map={"": 0},
        token=token,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(10, round(0.03 * len(train_ds) * args.epochs / args.effective_batch)),
        bf16=bf16_ok,
        fp16=not bf16_ok,
        max_length=args.max_length,
        completion_only_loss=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.epochs,
        report_to="none" if args.no_wandb else "wandb",
        run_name=f"qlora-r{args.lora_r}-script",
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if not bf16_ok:
        # fp16 grad scaler needs fp32 grads on the trainable params
        for p in trainer.model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()

    batch = next(iter(trainer.get_train_dataloader()))
    supervised = (batch["labels"] != -100).sum(dim=1)
    assert 1 <= int(supervised.min()) and int(supervised.max()) <= 2, (
        f"expected 1 or 2 supervised tokens per row, got {supervised.tolist()}"
    )
    print("masking verified: loss lands only on the level digit")

    has_checkpoint = bool(glob.glob(f"{args.output_dir}/checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if has_checkpoint else None)

    peft_model = trainer.model
    peft_model.config.use_cache = True
    scores: dict[str, float] = {}
    checkpoints = sorted(
        glob.glob(f"{args.output_dir}/checkpoint-*"),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    for ckpt in checkpoints:
        name = ckpt.rsplit("/", 1)[-1]
        peft_model.load_adapter(ckpt, adapter_name=name)
        peft_model.set_adapter(name)
        preds = predict_levels(peft_model, tokenizer, label_ids, val_df["sentence"].tolist(), 32)
        acc = accuracy_score(val_df["label"], preds)
        macro = f1_score(val_df["label"], preds, average="macro")
        scores[name] = float(macro)
        print(f"{name}: val accuracy {acc:.4f}, val macro F1 {macro:.4f}")

    best = max(scores, key=scores.get)
    print(f"best checkpoint: {args.output_dir}/{best} (val macro F1 {scores[best]:.4f})")
    print("next: python src/predict_finetuned.py --adapter", f"{args.output_dir}/{best}")


if __name__ == "__main__":
    main()
