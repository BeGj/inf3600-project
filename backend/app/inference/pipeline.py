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

# Set HF_INPAINT_LOCAL_ONLY=0 for the very first run (downloads the ~2GB base model).
LOCAL_FILES_ONLY = os.environ.get("HF_INPAINT_LOCAL_ONLY", "1") == "1"


class InpaintEngine:
    def __init__(self) -> None:
        self._pipe: StableDiffusionInpaintPipeline | None = None
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

    def infer(
        self,
        entry: ModelEntry,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int | None = None,
    ) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded. Did startup run?")

        image = image.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.Resampling.BILINEAR)
        mask = mask.convert("L").resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
        # Force a hard binary mask (white repaints) like the notebook does.
        mask = mask.point(lambda v: 255 if v > 127 else 0)

        if float((np.asarray(mask) > 0).mean()) == 0:
            raise ValueError("Mask is empty after rasterization; nothing to inpaint.")

        negative_prompt = entry.negative_prompt or None
        run_seed = seed if seed is not None else int.from_bytes(os.urandom(4), "big") % 2_147_483_647

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
                strength=STRENGTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            ).images[0]
        return result


# Module-level singleton.
engine = InpaintEngine()
