# CLAUDE.md

Guidance for AI agents and developers working in this repo. For full setup and the user
workflow, read [`README.md`](README.md) first; this file is the quick operational reference.

## What this is

A web app for **inpainting open satellite imagery**: search STAC catalogues (Sentinel-2 via
Earth Search) → stream a Cloud Optimized GeoTIFF (COG) onto an OpenLayers map → draw a
polygon → prompt → pick a model → get an inpainted patch back. Models are SD1.5-inpainting
+ LoRA adapters trained in `notebooks/`.

## Repo layout

| Dir          | Stack            | Entry points                                              |
|--------------|------------------|-----------------------------------------------------------|
| `frontend/`  | React/Vite/TS/OL | `src/App.tsx`, `src/api.ts` ([README](frontend/README.md))|
| `backend/`   | FastAPI + `uv`   | `app/main.py`, `app/routers/api.py` ([README](backend/README.md)) |
| `notebooks/` | diffusers/peft   | LoRA training + dataset prep ([README](notebooks/README.md)) |
| `scripts/`   | —                | `export_lora.py` (notebook adapter → `backend/models/`)   |

## Commands

```bash
# Frontend (cd frontend)
npm ci
npm run build      # tsc -b + vite build — THE type-check/CI gate. Run before committing TS changes.
npm run dev        # http://localhost:5173
npm run lint

# Backend (cd backend) — needs a CUDA GPU for real inference
uv sync
HF_INPAINT_LOCAL_ONLY=0 uv run fastapi dev   # first run downloads ~2GB base model
uv run fastapi dev                           # http://127.0.0.1:8000 ; docs at /docs
```

There is no backend test suite yet. To smoke-check backend modules without a GPU:
`python -m py_compile app/**/*.py` and the registry loads with stdlib only
(`python -c "from app.inference import registry; print(registry.list_models())"`).

## Architecture (request flow)

1. `GET /models` → registry entries (base + LoRAs) populate the UI dropdown.
2. `POST /catalog/search` (`app/catalog.py`) → `pystac-client` → Earth Search Sentinel-2
   L2A scenes, each with a `visual` (true-color) COG href.
3. Frontend streams the chosen scene's COG **directly** (browser → S3); it does not proxy
   through the backend.
4. `POST /inpaint` (`app/routers/api.py`): `geo.read_patch` reads the bbox window from the
   COG → 512² RGB; `geo.rasterize_mask` burns the polygon → 512² binary mask;
   `inference.pipeline.engine.infer` runs SD1.5-inpaint + the selected LoRA; returns a
   base64 PNG overlaid at the same bbox.

## Conventions & gotchas

- **Coordinates** crossing the API are **WGS-84 / EPSG:4326**. The map renders EPSG:3857;
  transform at the boundary only.
- **Mask convention**: white (255) = repaint, black = preserve. Kept binary end-to-end.
- **Inference params** (`backend/app/inference/pipeline.py`: `RESOLUTION`,
  `NUM_INFERENCE_STEPS`, `GUIDANCE_SCALE`, `STRENGTH`) mirror
  `notebooks/sanity_check_inpaint.ipynb` — keep them in sync so prod matches validation.
- **Context margin**: `/inpaint` reads a padded area around the polygon (`CONTEXT_MARGIN`
  / `INPAINT_CONTEXT_MARGIN`, default 0.5) so the model blends with real surroundings
  instead of generating a disconnected patch. The result is overlaid at the padded
  `context_bbox`, with the polygon as the alpha channel. If fills look disconnected,
  raise the margin; if context dwarfs the repaint region, lower it.
- **Inference is synchronous, in-process, lock-serialized** (one request at a time). Fine
  for a demo. For concurrency, move `engine.infer` behind a job queue + `/jobs/{id}` polling.
- **Models**: weights under `backend/models/<id>/` are **git-ignored**; only
  `backend/models/registry.json` is committed. Add a model by training it then running
  `scripts/export_lora.py` — it appears in `/models` after a backend restart, no code change.
- **All frontend backend calls** go through `frontend/src/api.ts`.
- **Catalogue registry** in `app/catalog.py` (`CATALOGS`): `sentinel-2` (`stac-api`, Earth
  Search, global 10 m) and `maxar` (`maxar-opendata`, ~0.5 m, disaster events only). Maxar is a
  *static* STAC catalogue handled in `app/maxar.py` via the opengeos per-event GeoJSON indexes
  (event-based: `GET /catalog/events` then search with `event`). Add a global STAC source by
  appending a `stac-api` `CatalogDef`; it auto-appears in the frontend via `GET /catalogs`.
  Non-true-color collections need band compositing before `geo.read_patch`.
- Python is pinned **3.12** (`backend/.python-version`) — the ML stack doesn't support 3.14.

## Git

`main` is the working branch. Generated artifacts (`notebooks/outputs/`, dataset contents, model
weights, `node_modules/`, `dist/`) are git-ignored — don't commit them.
