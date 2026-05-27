"""Singleton Stable Diffusion 1.5 inpainting pipeline with swappable LoRA adapters.

Inference parameters are ported verbatim from notebooks/sanity_check_inpaint.ipynb so the
production result matches what was validated during training:

    RESOLUTION = 512, NUM_INFERENCE_STEPS = 40, GUIDANCE_SCALE = 6.5, STRENGTH = 1.0
    mask convention: white (255) = repaint, black (0) = preserve.

The pipeline is built once at FastAPI startup. All LoRA adapters listed in the
registry are loaded once (each under its own adapter_name); a request selects which
adapter is active via set_adapters(). Inference is serialised with a lock because the
pipe is not reentrant and we run synchronously in-process.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from .registry import ModelEntry, list_models

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
RESOLUTION = 512
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 6.5
STRENGTH = 1.0

# FLUX.1-Fill-dev defaults. FLUX is guidance-distilled: no classical CFG (negative_prompt
# is ignored) and the embedded guidance runs much higher than SD1.5. FluxFillPipeline does
# not use `strength`. FLUX is ~12B params / ~24GB fp16, so it only runs on a big GPU.
FLUX_GUIDANCE_SCALE = 30.0
FLUX_NUM_INFERENCE_STEPS = 50
FLUX_MIN_VRAM_BYTES = 22 * 1024**3

# Set HF_INPAINT_LOCAL_ONLY=0 for the very first run (downloads the ~2GB base model, or
# ~24GB for FLUX). FLUX.1-Fill-dev is a gated repo: accept its license and set HF_TOKEN.
LOCAL_FILES_ONLY = os.environ.get("HF_INPAINT_LOCAL_ONLY", "1") == "1"


@lru_cache(maxsize=1)
def flux_available() -> tuple[bool, str | None]:
    """Whether the backend hardware/credentials can run FLUX.1-Fill-dev.

    Returns (True, None) when runnable, else (False, human-readable reason). The frontend
    uses the reason to grey out the FLUX option; the /inpaint handler uses it to reject a
    FLUX request defensively. Cached: hardware/env don't change within a process.
    """
    if not torch.cuda.is_available():
        return False, "FLUX (auto-disabled: backend has no CUDA GPU, requires 24GB+ VRAM)"
    total = torch.cuda.get_device_properties(0).total_memory
    if total < FLUX_MIN_VRAM_BYTES:
        gb = total / 1024**3
        return False, f"FLUX (auto-disabled: GPU has {gb:.0f}GB VRAM, requires 24GB+)"
    try:
        from huggingface_hub import get_token as _get_hf_token
        _hf_token = _get_hf_token()
    except Exception:
        _hf_token = None
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or _hf_token):
        return False, "FLUX (auto-disabled: set HF_TOKEN or run 'huggingface-cli login', and accept the FLUX.1-Fill-dev license)"
    return True, None


class InpaintEngine:
    def __init__(self) -> None:
        self._pipe: StableDiffusionInpaintPipeline | None = None
        self._flux_pipe = None  # FluxFillPipeline, lazy-loaded on first FLUX request
        self._loaded_adapters: set[str] = set()
        self._lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def load(self) -> None:
        """Build the base pipeline and register every LoRA adapter. Called at startup."""
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            safety_checker=None,
            local_files_only=LOCAL_FILES_ONLY,
        )
        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        for entry in list_models():
            adapter_dir = entry.adapter_dir
            if adapter_dir is None:
                continue
            if not adapter_dir.exists():
                # Manifest references an adapter whose weights were not shipped; skip
                # rather than crash so the rest of the API still works.
                print(f"[inpaint] WARNING: adapter dir missing for '{entry.id}': {adapter_dir}")
                continue
            pipe.load_lora_weights(str(adapter_dir), adapter_name=entry.id)
            self._loaded_adapters.add(entry.id)
            print(f"[inpaint] loaded LoRA adapter '{entry.id}'")

        self._pipe = pipe
        print(f"[inpaint] pipeline ready on {self.device} ({self.dtype})")

    def _select_adapter(self, entry: ModelEntry) -> None:
        assert self._pipe is not None
        if entry.type == "lora" and entry.id in self._loaded_adapters:
            self._pipe.set_adapters([entry.id], adapter_weights=[1.0])
        else:
            # Base model, or a LoRA whose weights weren't available: run with no adapter.
            try:
                self._pipe.disable_lora()
            except Exception:
                pass

    def _ensure_flux(self) -> None:
        """Lazy-load FLUX.1-Fill-dev on first use. Caller must hold self._lock.

        Loaded on demand (not at startup) so the ~24GB model isn't resident unless someone
        actually picks FLUX, and so boot stays fast for the common SD1.5 path.
        """
        if self._flux_pipe is not None:
            return
        ok, reason = flux_available()
        if not ok:
            raise RuntimeError(reason or "FLUX is unavailable on this backend.")
        from diffusers import FluxFillPipeline

        entry = next((e for e in list_models() if e.family == "flux-fill"), None)
        model_id = (entry.model_id if entry else None) or "black-forest-labs/FLUX.1-Fill-dev"
        print(f"[inpaint] loading FLUX pipeline '{model_id}' (first request, this is slow)…")
        pipe = FluxFillPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            local_files_only=LOCAL_FILES_ONLY,
        )
        pipe.enable_model_cpu_offload()
        self._flux_pipe = pipe
        print("[inpaint] FLUX pipeline ready")

    def infer(
        self,
        entry: ModelEntry,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int | None = None,
        negative_prompt: str | None = None,
        guidance_scale: float | None = None,
        strength: float | None = None,
        num_inference_steps: int | None = None,
    ) -> Image.Image:
        image = image.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.Resampling.BILINEAR)
        mask = mask.convert("L").resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
        # Force a hard binary mask (white repaints) like the notebook does.
        mask = mask.point(lambda v: 255 if v > 127 else 0)

        if float((np.asarray(mask) > 0).mean()) == 0:
            raise ValueError("Mask is empty after rasterization; nothing to inpaint.")

        run_seed = seed if seed is not None else int.from_bytes(os.urandom(4), "big") % 2_147_483_647

        if entry.family == "flux-fill":
            with self._lock:
                self._ensure_flux()
                # FLUX runs on CUDA (gated by flux_available); generator on the GPU.
                generator = torch.Generator("cuda").manual_seed(run_seed)
                # FLUX is guidance-distilled: no negative_prompt / strength support.
                return self._flux_pipe(
                    prompt=prompt or entry.default_prompt,
                    image=image,
                    mask_image=mask,
                    height=RESOLUTION,
                    width=RESOLUTION,
                    guidance_scale=(
                        guidance_scale if guidance_scale is not None else FLUX_GUIDANCE_SCALE
                    ),
                    num_inference_steps=(
                        num_inference_steps
                        if num_inference_steps is not None
                        else FLUX_NUM_INFERENCE_STEPS
                    ),
                    max_sequence_length=512,
                    generator=generator,
                ).images[0]

        # Default: SD1.5 + LoRA.
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded. Did startup run?")

        negative_prompt = negative_prompt or entry.negative_prompt or None

        with self._lock:
            self._select_adapter(entry)
            generator = torch.Generator(self.device).manual_seed(run_seed)
            result = self._pipe(
                prompt=prompt or entry.default_prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                height=RESOLUTION,
                width=RESOLUTION,
                strength=strength if strength is not None else STRENGTH,
                num_inference_steps=(
                    num_inference_steps if num_inference_steps is not None else NUM_INFERENCE_STEPS
                ),
                guidance_scale=guidance_scale if guidance_scale is not None else GUIDANCE_SCALE,
                generator=generator,
            ).images[0]
        return result


# Module-level singleton.
engine = InpaintEngine()
