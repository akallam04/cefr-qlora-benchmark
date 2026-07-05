"""Shared constants and helpers for the CEFR QLoRA benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
MANIFEST_PATH = REPO_ROOT / "data_manifest.json"

SEED = 42

LABEL_TO_CEFR = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}
CEFR_TO_LABEL = {v: k for k, v in LABEL_TO_CEFR.items()}
CEFR_LEVELS = [LABEL_TO_CEFR[i] for i in range(1, 7)]

# Single source of truth for the fine-tuned model's task format. The
# completion is one bare digit so classification is a single forward pass
# with an argmax over exactly six token logits: no generation, no parsing.
# Bare digit, not " digit": the Llama 3 tokenizer encodes " 1" as two
# tokens (space 220 + digit) but "1" as one token, verified against the
# released tokenizer (ids 16 to 21, stable after "Level:").
CLASSIFY_PROMPT = (
    "Rate the CEFR difficulty of this English sentence on a scale of "
    "1 (A1, easiest) to 6 (C2, hardest).\n"
    "Sentence: {sentence}\n"
    "Level:"
)


def format_prompt(sentence: str) -> str:
    """Fill the classification prompt for one sentence."""
    return CLASSIFY_PROMPT.format(sentence=sentence)


def completion_for(label: int) -> str:
    """Training completion for a gold label: one bare digit, one token."""
    return str(label)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting so reruns produce identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))
