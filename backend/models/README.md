# Backend models

The backend loads inpainting LoRA adapters listed in [`registry.json`](./registry.json).
`registry.json` **is committed**; the weight files are **not** (they are large) — they are
git-ignored and shipped via a mounted volume or release artifact.

## Layout

```
backend/models/
├── registry.json                  # committed manifest (model id -> prompt, adapter dir)
├── houses/
│   └── pytorch_lora_weights.safetensors   # git-ignored
└── trees/
    └── pytorch_lora_weights.safetensors   # git-ignored
```

The `base` entry needs no weights (runs the SD1.5 inpainting model with no adapter).

## Exporting a trained adapter

Training notebooks (`notebooks/train_houses_lora.ipynb`, `train_trees_lora.ipynb`) produce
adapters under `notebooks/outputs/lora_<name>/`. Normalise one
into this directory and register it with `scripts/export_lora.py` (run from the notebooks
env, which has torch/diffusers/peft):

```bash
cd notebooks
uv run python ../scripts/export_lora.py \
  --id houses --label "Houses" \
  --adapter outputs/lora_houses \
  --prompt "satellite view of small houses, roof geometry, driveways, residential block" \
  --negative "blurry, distorted, repeated roofs, warped perspective, low quality"

uv run python ../scripts/export_lora.py \
  --id trees --label "Trees" \
  --adapter outputs/lora_trees \
  --prompt "satellite view of trees, canopy cover, urban vegetation" \
  --negative "blurry, distorted, low quality, cartoon, warped perspective, repeated artifacts"
```

The script writes `backend/models/<id>/pytorch_lora_weights.safetensors` and upserts the
`registry.json` entry. Restart the backend to pick up new/changed weights (adapters load at
startup).
