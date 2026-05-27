"""Model registry: reads backend/models/registry.json and resolves model ids.

The registry is the single source of truth for which models the API exposes.
Each entry is one of:

    {"id": "base", "label": "Base (no LoRA)", "type": "base",
     "default_prompt": "...", "negative_prompt": "..."}

    {"id": "houses", "label": "Houses", "type": "lora",
     "adapter_path": "houses",   # dir under backend/models/ holding the .safetensors
     "default_prompt": "...", "negative_prompt": "..."}

LoRA weights themselves are git-ignored and shipped via volume/release; only this
JSON manifest is committed. See backend/models/README.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# backend/app/inference/registry.py -> backend/
BACKEND_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = BACKEND_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"


@dataclass(frozen=True)
class ModelEntry:
    id: str
    label: str
    type: str  # "base" | "lora"
    default_prompt: str = ""
    negative_prompt: str = ""
    adapter_path: str | None = None
    # Which inference backend runs this entry: "sd15" (StableDiffusionInpaintPipeline +
    # LoRA) or "flux-fill" (FluxFillPipeline). Drives engine dispatch and which inference
    # params the frontend exposes.
    family: str = "sd15"
    # HF repo id for non-sd15 families (e.g. "black-forest-labs/FLUX.1-Fill-dev").
    model_id: str | None = None

    @property
    def adapter_dir(self) -> Path | None:
        """Absolute path to the directory holding this LoRA's safetensors."""
        if self.type != "lora" or not self.adapter_path:
            return None
        return MODELS_DIR / self.adapter_path

    def public(self, available: bool = True, disabled_reason: str | None = None) -> dict:
        """Fields safe to expose to the frontend via GET /models.

        `available`/`disabled_reason` are computed at request time (see the /models
        handler) because availability depends on backend hardware, not the manifest.
        """
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "family": self.family,
            "default_prompt": self.default_prompt,
            "negative_prompt": self.negative_prompt,
            "available": available,
            "disabled_reason": disabled_reason,
        }


@lru_cache(maxsize=1)
def load_registry() -> dict[str, ModelEntry]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Model registry not found at {REGISTRY_PATH}. "
            "Run scripts/export_lora.py to populate backend/models/."
        )
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    entries: dict[str, ModelEntry] = {}
    for item in raw.get("models", []):
        entry = ModelEntry(
            id=item["id"],
            label=item.get("label", item["id"]),
            type=item.get("type", "lora"),
            default_prompt=item.get("default_prompt", ""),
            negative_prompt=item.get("negative_prompt", ""),
            adapter_path=item.get("adapter_path"),
            family=item.get("family", "sd15"),
            model_id=item.get("model_id"),
        )
        entries[entry.id] = entry
    return entries


def list_models() -> list[ModelEntry]:
    return list(load_registry().values())


def get_model(model_id: str) -> ModelEntry:
    registry = load_registry()
    if model_id not in registry:
        raise KeyError(model_id)
    return registry[model_id]
