#!/usr/bin/env python
"""Export a trained LoRA adapter into the backend's models directory.

Training produces adapters in two shapes:
  * peft format    — adapter_model.safetensors + adapter_config.json
                     (e.g. notebooks/outputs/lora-satellite-inpaint/lora-final/)
  * diffusers fmt  — pytorch_lora_weights.safetensors
                     (e.g. notebooks/outputs/lora_houses/)

This script normalises either into the diffusers format the backend loads, writing
    backend/models/<id>/pytorch_lora_weights.safetensors
and upserting the matching entry in backend/models/registry.json.

Usage:
    uv run python scripts/export_lora.py \
        --id houses --label "Houses" \
        --adapter notebooks/outputs/lora_houses \
        --prompt "satellite view of small houses, roof geometry, driveways, residential block" \
        --negative "blurry, distorted, repeated roofs, warped perspective, low quality"

Run it inside the notebooks env (which has torch/diffusers/peft installed):
    cd notebooks && uv run python ../scripts/export_lora.py ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffusers import StableDiffusionInpaintPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "backend" / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"


def export_weights(adapter_dir: Path, out_dir: Path) -> None:
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "pytorch_lora_weights.safetensors"

    # If already in diffusers format, copy directly — no pipeline round-trip needed.
    diffusers_weights = adapter_dir / "pytorch_lora_weights.safetensors"
    if diffusers_weights.exists():
        shutil.copy2(diffusers_weights, dest)
        print(f"[export] wrote {dest}")
        return

    # peft format (adapter_model.safetensors + adapter_config.json): load into
    # pipeline so diffusers can re-key the tensors, then extract the UNet state dict.
    import torch
    from peft import get_peft_model_state_dict

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, safety_checker=None
    )
    pipe.load_lora_weights(str(adapter_dir))
    unet_lora_state_dict = get_peft_model_state_dict(pipe.unet)
    StableDiffusionInpaintPipeline.save_lora_weights(
        save_directory=str(out_dir),
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    print(f"[export] wrote {dest}")


def upsert_registry(model_id: str, label: str, prompt: str, negative: str) -> None:
    if REGISTRY_PATH.exists():
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        data = {"models": []}

    entry = {
        "id": model_id,
        "label": label,
        "type": "lora",
        "adapter_path": model_id,
        "default_prompt": prompt,
        "negative_prompt": negative,
    }
    models = [m for m in data.get("models", []) if m.get("id") != model_id]
    models.append(entry)
    data["models"] = models
    REGISTRY_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[export] updated registry entry '{model_id}'")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", required=True, help="Model id used by the API and adapter dir name.")
    p.add_argument("--label", required=True, help="Human-readable name shown in the UI.")
    p.add_argument("--adapter", required=True, type=Path, help="Trained adapter directory.")
    p.add_argument("--prompt", default="", help="Default prompt for this model.")
    p.add_argument("--negative", default="", help="Default negative prompt.")
    args = p.parse_args()

    if not args.adapter.exists():
        raise SystemExit(f"Adapter directory not found: {args.adapter}")

    export_weights(args.adapter, MODELS_DIR / args.id)
    upsert_registry(args.id, args.label, args.prompt, args.negative)
    print("[export] done.")


if __name__ == "__main__":
    main()
