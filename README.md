# INF-3600 Project

## Backend
requires uv. Boilerplate setup based on https://github.com/astral-sh/uv-fastapi-example 
- `uv run fastapi dev`
- http://127.0.0.1:8000/docs


## frontend
requires node lts and npm
- `npm ci`
- `npm run dev`
- http://localhost:5173/
## experiment
Requires uv.
- Various python scripts for experimenting

## Datasets
Prepared datasets are stored under `datasets/`. The generated image, mask, and metadata files are ignored by Git.

From `notebooks/`, run:

```powershell
uv sync
uv run python scripts/prepare_morocco_buildings_lora_dataset.py --output-dir ../datasets/houses
uv run python scripts/prepare_wroclaw_trees_lora_dataset.py --output-dir ../datasets/trees
```
