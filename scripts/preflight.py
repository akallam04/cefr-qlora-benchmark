"""Preflight checks for the CEFR QLoRA benchmark.

Verifies that every external service this project depends on is reachable
with the credentials in .env before any real work starts:

1. OpenAI API key is valid and gpt-4o-mini is available (baseline).
2. Hugging Face token is valid and has access to the gated
   meta-llama/Meta-Llama-3-8B repository (Colab fine-tune).
3. Weights & Biases API key is valid (training telemetry).

Usage:
    .venv/bin/python scripts/preflight.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

GATED_MODEL = "meta-llama/Meta-Llama-3-8B"
REQUIRED_KEYS = ("OPENAI_API_KEY", "HF_TOKEN", "WANDB_API_KEY")


def check_env_present() -> dict[str, str]:
    """Load .env and confirm all required keys are set. Exits if any are missing."""
    load_dotenv()
    values: dict[str, str] = {}
    missing: list[str] = []
    for key in REQUIRED_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            values[key] = value
        else:
            missing.append(key)
    if missing:
        print(f"FAIL .env is missing values for: {', '.join(missing)}")
        print("     Copy .env.example to .env and paste your keys.")
        sys.exit(1)
    print("PASS .env contains all three keys")
    return values


def check_openai(api_key: str) -> bool:
    """Confirm the OpenAI key works using a free metadata call."""
    from openai import OpenAI

    try:
        models = OpenAI(api_key=api_key).models.list()
        available = {m.id for m in models}
    except Exception as exc:
        print(f"FAIL OpenAI key rejected: {exc}")
        return False
    if "gpt-4o-mini" not in available:
        print("FAIL OpenAI key works but gpt-4o-mini is not available to this account")
        return False
    print("PASS OpenAI key valid, gpt-4o-mini available")
    return True


def check_huggingface(token: str) -> bool:
    """Confirm the HF token works and can reach the gated Llama 3 repo."""
    from huggingface_hub import auth_check, whoami
    from huggingface_hub.errors import GatedRepoError

    try:
        user = whoami(token=token)["name"]
    except Exception as exc:
        print(f"FAIL Hugging Face token rejected: {exc}")
        return False
    print(f"PASS Hugging Face token valid (logged in as {user})")

    try:
        auth_check(GATED_MODEL, token=token)
    except GatedRepoError:
        print(f"FAIL No access to {GATED_MODEL} yet.")
        print(f"     Request it at https://huggingface.co/{GATED_MODEL} and wait for approval.")
        return False
    except Exception as exc:
        print(f"FAIL Could not verify gated access: {exc}")
        return False
    print(f"PASS Gated access confirmed for {GATED_MODEL}")
    return True


def check_wandb(api_key: str) -> bool:
    """Confirm the W&B key is accepted by the server."""
    import wandb

    try:
        ok = wandb.login(key=api_key, verify=True, relogin=True)
    except Exception as exc:
        print(f"FAIL Weights & Biases key rejected: {exc}")
        return False
    if not ok:
        print("FAIL Weights & Biases login returned False")
        return False
    print("PASS Weights & Biases key valid")
    return True


def main() -> None:
    """Run every check and exit nonzero if any fail."""
    values = check_env_present()
    results = [
        check_openai(values["OPENAI_API_KEY"]),
        check_huggingface(values["HF_TOKEN"]),
        check_wandb(values["WANDB_API_KEY"]),
    ]
    if all(results):
        print("\nAll preflight checks passed. Ready to build.")
    else:
        print("\nSome checks failed. Fix the FAIL lines above and rerun.")
        sys.exit(1)


if __name__ == "__main__":
    main()
