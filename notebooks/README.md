# Notebooks

Experimentation and training notebooks for satellite-image inpainting. All
notebooks use the [`uv`](https://docs.astral.sh/uv/) environment defined in
`pyproject.toml`. Run `uv sync` once, then launch Jupyter (or open the notebooks
in VS Code) with the project's `.venv` kernel.

```powershell
uv sync
uv run jupyter lab
```

GPU is assumed (CUDA 12.8 wheels are pinned in `pyproject.toml`). Generated
images, masks, dataset files, and trained weights are git-ignored.

### Notebook output stripping

Cell outputs are stripped from notebooks before commit so generated content
never bloats git history. This is enforced by a [`nbstripout`](https://github.com/kynan/nbstripout)
pre-commit hook (`.pre-commit-config.yaml`). Enable it once per clone:

```bash
uv tool install pre-commit
pre-commit install
```

`.gitattributes` also registers `nbstripout` as a git filter for clean diffs; to
activate that locally run `uv tool install nbstripout && nbstripout --install`.

## Dataset preparation

The training notebooks read prepared datasets from `../datasets/`. Generate them
first with the scripts under `scripts/`:

```powershell
uv run python scripts/prepare_morocco_buildings_lora_dataset.py --output-dir ../datasets/houses
uv run python scripts/prepare_wroclaw_trees_lora_dataset.py --output-dir ../datasets/trees
```

Each script writes images, masks, and a `metadata.jsonl` manifest.

## Notebooks

### Training

| Notebook | Purpose |
| --- | --- |
| `train_houses_lora.ipynb` | Self-contained LoRA training for `stable-diffusion-v1-5/stable-diffusion-inpainting`. Reads `datasets/houses` and saves website-ready adapter weights to `outputs/lora_houses`. Uses 🤗 `accelerate` and locates the repo root automatically. |
| `train_trees_lora.ipynb` | Same pipeline as the houses notebook, but trains on `datasets/trees` and writes to `outputs/lora_trees`. |
| `002_lora_finetune.ipynb` | Earlier LoRA fine-tuning experiment. Fine-tunes SD2 inpainting on the [`arampacha/rsicd`](https://huggingface.co/datasets/arampacha/rsicd) satellite caption dataset, training only ~3 MB of adapter weights. Hyperparameters are collected in a single `CFG` dict. Tested on an RTX 3080 (10 GB), ~1–2 h for 3 epochs. Superseded by the `train_*_lora` notebooks. |

### Inference / testing

| Notebook | Purpose |
| --- | --- |
| `test_notebook.ipynb` | Minimal sanity check for SD 1.5 inpainting. Deliberately ignores the rest of the repo: loads one image + one mask from a prepared dataset and runs the base `stable-diffusion-v1-5/stable-diffusion-inpainting` model — no LoRA, backend, or crop logic. Useful for confirming an image/mask pair is correct (white pixels are repainted). |
| `001_sd-inpaint.ipynb` | Interactive Stable Diffusion 3 inpainting playground. Upload an image with an `ipywidgets` uploader, draw a polygon mask on an `ipycanvas` canvas, set prompt / negative prompt, and run the `StableDiffusion3InpaintPipeline`. Saves output to `inpaint_result.png`. |
| `003_test-inpaint.ipynb` | Empty placeholder (no cells yet). |
| `qwen-image-edit.ipynb` | Experiment with the `QwenImageEditPipeline` for instruction-based image editing, using 4-bit (bitsandbytes) quantization and an `ipywidgets` upload UI. Not part of the inpainting pipeline; kept for comparison. |

## Conventions

- Masks are binary; **white pixels mark the region the model repaints**.
- The training notebooks find the repo root by walking up to the directory that
  contains both `backend/` and `frontend/`, so they can be run from anywhere in
  the tree.
- Trained adapters land in `outputs/lora_<name>/` (diffusers layout); peft-format
  checkpoints under `lora-satellite-inpaint/` work too.

## Deploying a trained model to the backend

Training only produces the adapter; the backend serves it. Normalise an adapter into
`backend/models/` and register it with [`scripts/export_lora.py`](../scripts/export_lora.py):

```bash
uv run python ../scripts/export_lora.py \
  --id trees --label "Trees" \
  --adapter ../outputs/lora_trees \
  --prompt "satellite view of trees, canopy cover, urban vegetation" \
  --negative "blurry, distorted, low quality, cartoon, warped perspective, repeated artifacts"
```

This writes `backend/models/<id>/pytorch_lora_weights.safetensors` (git-ignored) and updates
the committed `backend/models/registry.json`. Restart the backend to pick it up — it then
appears in `GET /models` and the UI dropdown automatically. See
[`backend/models/README.md`](../backend/models/README.md).
