# INF-3600 — Satellite Generative Fill

A web app for **inpainting open satellite imagery**. Users search free/open STAC
catalogues (Sentinel-2 via Earth Search), stream the selected Cloud Optimized GeoTIFF
(COG) onto a map, draw a polygon over an area, type what should appear there, pick a
model, and get back an inpainted patch overlaid in place.

```
┌─────────────┐   /catalog/search   ┌──────────────┐  pystac-client   ┌──────────────┐
│  frontend   │ ──────────────────▶ │   backend    │ ───────────────▶ │ Earth Search │
│ React + OL  │ ◀── scenes (COG) ── │   FastAPI    │                  │  (STAC API)  │
│             │                     │              │                  └──────────────┘
│  draw mask  │     /inpaint        │ rasterio COG │   SD1.5-inpaint + LoRA (diffusers)
│  + prompt   │ ──────────────────▶ │ + pipeline   │ ──▶ PNG patch ──▶ map overlay
└─────────────┘                     └──────────────┘
        │ streams the selected scene's COG directly (browser → S3) via OpenLayers
        └──────────────────────────────────────────────────────────────────────▶
```

Three parts of the repo:

| Dir          | What it is                                                              |
|--------------|-------------------------------------------------------------------------|
| `frontend/`  | React 19 + Vite + OpenLayers map UI.                                    |
| `backend/`   | FastAPI service: catalogue search + COG reading + inpainting inference. |
| `notebooks/` | LoRA training + dataset prep ([details](notebooks/README.md)).          |

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (Python tooling) and **Python 3.12**.
- Node.js LTS + npm.
- A **CUDA GPU** for usable inference latency (CPU runs but is slow).
- Trained LoRA weights exported into `backend/models/` (see step 1).

## Get started

### 1. Export models into the backend

Trained adapters from the notebooks must be normalised into `backend/models/`. From the
notebooks env (which has torch/diffusers/peft):

```bash
cd notebooks
uv sync
uv run python ../scripts/export_lora.py \
  --id houses --label "Houses" \
  --adapter lora-satellite-inpaint/lora-final \
  --prompt "satellite view of small houses, roof geometry, driveways, residential block" \
  --negative "blurry, distorted, repeated roofs, warped perspective, low quality"

uv run python ../scripts/export_lora.py \
  --id trees --label "Trees" \
  --adapter ../outputs/lora_trees \
  --prompt "satellite view of trees, canopy cover, urban vegetation" \
  --negative "blurry, distorted, low quality, cartoon, warped perspective, repeated artifacts"
```

This writes git-ignored weights to `backend/models/<id>/` and updates the committed
`backend/models/registry.json`. The `base` (no-LoRA) model needs no weights. See
[`backend/models/README.md`](backend/models/README.md).

> You can skip this and run with only the **Base** model — the backend logs a warning and
> skips any LoRA whose weights are missing.

### 2. Run the backend

```bash
cd backend
uv sync
HF_INPAINT_LOCAL_ONLY=0 uv run fastapi dev   # first run: downloads ~2GB base model
# later runs:  uv run fastapi dev
```

- API: http://127.0.0.1:8000 · interactive docs: http://127.0.0.1:8000/docs
- The pipeline + all registry LoRAs load at **startup** (first request is not delayed).

### 3. Run the frontend

```bash
cd frontend
npm ci
npm run dev        # http://localhost:5173
```

### 4. Use it

1. Pan/zoom the map to your area of interest.
2. **Find imagery** → set date range + max cloud cover → **Search this view** → pick a scene.
3. Choose a **Model**, **Draw Mask** over a region, edit the **Prompt**, then **Generate**.
4. The inpainted patch overlays at the correct geographic extent; **Download Last Patch** saves it.

## Configuration

| Where    | Variable               | Default                              | Purpose                                    |
|----------|------------------------|--------------------------------------|--------------------------------------------|
| backend  | `HF_INPAINT_LOCAL_ONLY`| `1`                                  | Set `0` on first run to download the base model. |
| backend  | `ALLOWED_ORIGINS`      | `http://localhost:5173,…127.0.0.1…`  | Comma-separated CORS origins.              |
| backend  | `INPAINT_CONTEXT_MARGIN`| `0.5`                               | Extra imagery read around the polygon (per side, as a fraction of its bbox) for the model to blend with. Higher = more context, smaller repaint region in-frame. |
| frontend | `VITE_API_URL`         | `http://localhost:8000`              | Backend base URL (set in `frontend/.env`). |

## API reference

| Method | Path              | Body / params                                              | Returns                       |
|--------|-------------------|------------------------------------------------------------|-------------------------------|
| GET    | `/models`         | —                                                          | `[{id,label,type,default_prompt}]` |
| GET    | `/catalogs`       | —                                                          | `[{id,label,resolution_m,coverage,supports_cloud,supports_datetime,requires_event}]` |
| GET    | `/catalog/events` | `?catalog=maxar`                                           | `[{id,label}]` (event-based catalogues) |
| POST   | `/catalog/search` | `{bbox,catalog,event?,datetime?,limit?,max_cloud_cover?}`  | `[{id,datetime,cloud_cover,bbox,visual_href,thumbnail}]` |
| POST   | `/inpaint`        | `{image_url,bbox,mask_geojson,prompt,model_id,seed?}`      | `{image_b64,bbox}`            |

Full schemas at `/docs`.

## Developing further

**Project layout (backend):**

```
backend/app/
├── main.py                  # app + CORS + startup (loads the pipeline)
├── routers/api.py           # /models, /catalog/search, /inpaint
├── catalog.py               # STAC search (Earth Search)
├── geo.py                   # COG window read + mask rasterization
└── inference/
    ├── registry.py          # reads models/registry.json
    └── pipeline.py          # singleton SD inpaint pipeline + LoRA adapters
```

Common extensions:

- **Add a new LoRA model** — train it (see [`notebooks/README.md`](notebooks/README.md)),
  then run `scripts/export_lora.py --id <name> ...`. It appears in `/models` and the UI
  dropdown automatically after a backend restart. No code changes needed.
- **Add another imagery catalogue** — `backend/app/catalog.py` holds a `CATALOGS` registry.
  For a standard STAC API, add one `CatalogDef` with `kind="stac-api"` (set `stac_url`,
  `collections`, `asset_key`) — it shows up in the frontend dropdown automatically via
  `GET /catalogs`. Non-STAC/static sources (like Maxar Open Data, see `app/maxar.py`) use a
  custom `kind` with its own handler. All sources return the same scene dict shape (the
  frontend keys off `visual_href` + `bbox`). Non-true-color collections need a band-
  compositing step before `geo.read_patch` returns RGB.
- **Tune inference** — generation params live as constants in
  `backend/app/inference/pipeline.py` (`RESOLUTION`, `NUM_INFERENCE_STEPS`,
  `GUIDANCE_SCALE`, `STRENGTH`). They mirror `notebooks/test_notebook.ipynb`; keep the two
  in sync so prod matches what you validate while training.
- **Heavier scale** — inference is **synchronous, in-process, and lock-serialized** (one
  request at a time), which is fine for a demo/single user. To support concurrency, move
  `engine.infer` behind a job queue (e.g. a worker process + `/jobs/{id}` polling) and have
  the frontend poll instead of awaiting `/inpaint` directly.
- **Frontend** — `frontend/src/`:
  `CatalogPanel.tsx` (search), `PromptPanel.tsx` (model/prompt/mask controls),
  `MapView.tsx` (OpenLayers map, COG streaming, polygon draw, result overlays),
  `App.tsx` (state wiring), `api.ts` (typed backend client). `npm run build` runs the
  TypeScript type-check.

## Datasets & training

Dataset prep and LoRA training live in `notebooks/` — see [`notebooks/README.md`](notebooks/README.md).
Generated images, masks, dataset files, model weights, and `outputs/` are git-ignored.
