# Notebooks

Experimentation and training notebooks for satellite-image inpainting. All
notebooks use the [`uv`](https://docs.astral.sh/uv/) environment defined in
`pyproject.toml`. Run `uv sync` once, then launch Jupyter (or open the notebooks
in VS Code) with the project's `.venv` kernel.

```powershell
uv sync
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

## Layout

```
notebooks/
  prepare/                          data-prep scripts (source dataset -> image/mask pairs)
    prepare_houses.py
    prepare_trees.py
    prepare_osm_sentinel.py
  train_houses_lora.ipynb           single-class LoRA training notebooks
  train_trees_lora.ipynb
  train_sentinel_water_lora.ipynb
  train_small_house_set_lora.ipynb
  train_small_tree_set_lora.ipynb
  train_osm_lora.py                 multi-class Nordic LoRA training (script)
  sanity_check_inpaint.ipynb        base-model inpaint sanity checks
  sanity_check_inpaint_split.ipynb
  fid_base_vs_lora_inpaint.ipynb    base vs LoRA FID evaluation
  fid_base_vs_lora_inpaint_osm_lora.ipynb
  exploration/                      kept-for-reference experiments (not the active pipeline)
  outputs/                          trained adapters + sample renders (git-ignored)
```

Two kinds of artifacts live here: **`.ipynb` notebooks** for interactive training and
evaluation, and a couple of **`.py` scripts** (`prepare/*.py`, `train_osm_lora.py`) for
longer non-interactive runs you launch with `uv run python ...`.

## Dataset preparation

The training notebooks read prepared datasets from `../datasets/`. Generate them
first with the scripts under `prepare/`. Each script writes images, masks, and a
`metadata.jsonl` manifest in the same on-disk format.

```powershell
# Single-class HF-sourced datasets
uv run python prepare/prepare_houses.py --output-dir ../datasets/houses
uv run python prepare/prepare_trees.py  --output-dir ../datasets/trees

# Multi-class Nordic dataset from live Sentinel-2 + ESA WorldCover land cover
uv run python prepare/prepare_osm_sentinel.py --output-dir ../datasets/osm_nordic
```

| Script                    | Source                                                                                                                                                                                                               | Output                |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `prepare_houses.py`       | HF dataset → image/mask pairs                                                                                                                                                                                        | `datasets/houses`     |
| `prepare_trees.py`        | HF dataset → image/mask pairs                                                                                                                                                                                        | `datasets/trees`      |
| `prepare_osm_sentinel.py` | Samples Nordic tiles, reads ESA WorldCover 2021 land cover (free 10 m COG) to label/mask, pulls cloud-free Sentinel-2 patches from Earth Search. 9 land-cover classes, each record carries its own per-class prompt. | `datasets/osm_nordic` |

## Notebooks & scripts

### Training

All training targets `stable-diffusion-v1-5/stable-diffusion-inpainting`, uses 🤗
`accelerate`, locates the repo root automatically, and writes adapters under
`notebooks/outputs/`.

| File                               | Reads                      | Writes                                      | Notes                                                                                                                                                                                                               |
| ---------------------------------- | -------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_houses_lora.ipynb`          | `datasets/houses`          | `outputs/lora_houses`                       | Self-contained single-class LoRA training.                                                                                                                                                                          |
| `train_trees_lora.ipynb`           | `datasets/trees`           | `outputs/lora_trees`                        | Same pipeline, trees class.                                                                                                                                                                                         |
| `train_sentinel_water_lora.ipynb`  | `datasets/sentinel_water`  | `outputs/lora_sentinel_water_sd15_inpaint`  | Water class on Sentinel-2 patches.                                                                                                                                                                                  |
| `train_small_house_set_lora.ipynb` | `datasets/small_house_set` | `outputs/lora_small_house_set_sd15_inpaint` | Small-dataset house variant.                                                                                                                                                                                        |
| `train_small_tree_set_lora.ipynb`  | `datasets/small_tree_set`  | `outputs/lora_small_tree_set_sd15_inpaint`  | Small-dataset tree variant.                                                                                                                                                                                         |
| `train_osm_lora.py` (script)       | `datasets/osm_nordic`      | `outputs/`                                  | Multi-class Nordic LoRA. Per-class prompts, cosine LR (100-step warmup + 3000 steps), mask-weighted loss, flip/rotation augmentation, holds out the last record per class for validation. Logs to Weights & Biases. |

### Inference / evaluation

| File                                      | Purpose                                                                                                                                                                                                                                                                                |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sanity_check_inpaint.ipynb`              | Minimal sanity check for SD 1.5 inpainting. Deliberately ignores the rest of the repo: loads one image + one mask from a prepared dataset and runs the base model — no LoRA, backend, or crop logic. Useful for confirming an image/mask pair is correct (white pixels are repainted). |
| `sanity_check_inpaint_split.ipynb`        | Same idea, but keeps model loading separate from inference settings — load the pipeline once, then tweak prompt / seed / strength / guidance / steps / LoRA scale in the settings cells and rerun only the inference cell.                                                             |
| `fid_base_vs_lora_inpaint.ipynb`          | Runs the same masked inputs through the base model and the base + a selected LoRA, then computes FID for each generated set against the original dataset images (lower is better). Bump `NUM_SAMPLES` for a reliable score; defaults are small for a fast first run.                   |
| `fid_base_vs_lora_inpaint_osm_lora.ipynb` | Same FID comparison wired up for the Nordic OSM LoRA / dataset.                                                                                                                                                                                                                        |

### Exploration (`exploration/`)

Kept for reference; not part of the active inpainting pipeline.

| Notebook                   | Purpose                                                                                                                                                                                                                                                                                                         |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sd2_lora_finetune.ipynb`  | Earlier LoRA fine-tuning experiment. Fine-tunes SD2 inpainting on the [`arampacha/rsicd`](https://huggingface.co/datasets/arampacha/rsicd) satellite caption dataset, training only ~3 MB of adapter weights. Hyperparameters are collected in a single `CFG` dict. Superseded by the `train_*_lora` notebooks. |
| `explore_sd_inpaint.ipynb` | Interactive Stable Diffusion 3 inpainting playground. Upload an image with an `ipywidgets` uploader, draw a polygon mask on an `ipycanvas` canvas, set prompt / negative prompt, and run the `StableDiffusion3InpaintPipeline`.                                                                                 |
| `explore_qwen_edit.ipynb`  | Experiment with the `QwenImageEditPipeline` for instruction-based image editing, using 4-bit (bitsandbytes) quantization and an `ipywidgets` upload UI.                                                                                                                                                         |

## Conventions

- Masks are binary; **white pixels mark the region the model repaints**.
- The training notebooks find the repo root by walking up to the directory that
  contains `backend/`, `frontend/`, and `notebooks/`, so they can be run from
  anywhere in the tree.
- Trained adapters land in `notebooks/outputs/lora_<name>/` (diffusers layout);
  peft-format checkpoints (e.g. `notebooks/outputs/lora-satellite-inpaint/`) work too.

## Deploying a trained model to the backend

Training only produces the adapter; the backend serves it. Normalise an adapter into
`backend/models/` and register it with [`scripts/export_lora.py`](../scripts/export_lora.py):

```bash
uv run python ../scripts/export_lora.py \
  --id trees --label "Trees" \
  --adapter outputs/lora_trees \
  --prompt "satellite view of trees, canopy cover, urban vegetation" \
  --negative "blurry, distorted, low quality, cartoon, warped perspective, repeated artifacts"
```

This writes `backend/models/<id>/pytorch_lora_weights.safetensors` (git-ignored) and updates
the committed `backend/models/registry.json`. Restart the backend to pick it up — it then
appears in `GET /models` and the UI dropdown automatically. See
[`backend/models/README.md`](../backend/models/README.md).
