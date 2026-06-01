# Backend — Satellite Generative Fill

FastAPI service that streams Sentinel-2 imagery metadata from open STAC catalogues and
runs Stable Diffusion 1.5 inpainting (with optional LoRA adapters) over user-drawn regions.

## Endpoints

| Method | Path              | Purpose                                                                |
| ------ | ----------------- | ---------------------------------------------------------------------- |
| GET    | `/`               | Health check (`{"status": "ok", ...}`).                                |
| GET    | `/models`         | List selectable models (base + trained LoRAs + FLUX) from the registry. |
| GET    | `/catalogs`       | List imagery catalogues + metadata (resolution, coverage, flags).      |
| GET    | `/catalog/events` | List events for an event-based catalogue (`?catalog=maxar`).           |
| POST   | `/catalog/search` | Search a catalogue by `catalog`/`event`/bbox/date/cloud cover.         |
| POST   | `/inpaint`        | Read a COG patch, rasterize the mask, run inpainting (see below).      |
| GET    | `/jobs/{job_id}`  | Poll a long-running (FLUX) inpaint job's status/result.                |

`/inpaint` has two response shapes depending on the model family: **SD1.5/LoRA** runs
synchronously and returns the PNG directly (`{image_b64, bbox}`). **FLUX** can block for a
long time (cold model download + slow inference), so it returns `202 Accepted` with a
`{job_id}` immediately; the client then polls `/jobs/{job_id}` until `status` is `done`
(`result_b64`/`result_bbox` populated) or `error`. Jobs live in an in-process store
(`app/jobs.py`) served by a single-worker thread pool, since the GPU runs one inference at
a time. The returned PNG uses the (feathered) polygon mask as its alpha channel so the
overlay shows generated content only inside the drawn region.

## Catalogues

`app/catalog.py` holds a registry (`CATALOGS`) of imagery sources, each a `CatalogDef`:

- **`sentinel-2`** (`stac-api`) — Earth Search Sentinel-2 L2A, 10 m, global, queried with
  pystac-client (bbox + datetime + cloud).
- **`maxar`** (`maxar-opendata`) — Maxar Open Data, ~0.5 m, *disaster-event areas only*. It is a
  *static* STAC catalogue (no `/search`), so `app/maxar.py` discovers events and tiles via the
  community [opengeos/maxar-open-data](https://github.com/opengeos/maxar-open-data) per-event
  GeoJSON indexes, filtering tiles by bbox/cloud with `shapely`. Public COGs (no signing).

Add another global STAC source by appending a `stac-api` `CatalogDef` (set `stac_url`,
`collections`, `asset_key`) — no other code changes needed.

## Models

Trained LoRA weights live under `models/<id>/` (git-ignored) and are described by the
committed `models/registry.json`. See [`models/README.md`](./models/README.md) for how to
export adapters from the training notebooks via `scripts/export_lora.py`.

### FLUX.1-Fill-dev

The registry also lists `flux-fill` (`family: "flux-fill"`), which runs Black Forest Labs'
**FLUX.1-Fill-dev** via diffusers' `FluxFillPipeline` (in `bfloat16`) instead of SD1.5. It
is much larger (~12B params) and has prerequisites the other models don't:

- **GPU**: a CUDA device with **≥ 24 GB VRAM** (the gate in `app/inference/pipeline.py`,
  `FLUX_MIN_VRAM_BYTES`, requires ≥ 22 GB).
- **Gated model**: accept the license at
  <https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev> and provide a token via
  `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) / `huggingface-cli login`.

When any prerequisite is missing, `GET /models` reports the FLUX entry as
`available: false` with a `disabled_reason`, the frontend greys out the option, and
`/inpaint` rejects a direct FLUX request with `503`. FLUX is **lazy-loaded on the first
FLUX request** (not at startup), so the first generation is slow (and downloads ~34 GB if
not cached). FLUX is guidance-distilled, so it ignores `negative_prompt`/`strength` and
defaults to `guidance_scale=30`, `num_inference_steps=50`. The existing SD1.5 LoRAs are
**not** compatible with it.

## Running

Requires a CUDA GPU for usable latency (CPU works but is slow).

```bash
cd backend
uv sync
# First run downloads the ~2GB base SD1.5-inpainting model:
HF_INPAINT_LOCAL_ONLY=0 uv run fastapi dev
# Subsequent runs can use the cache:
uv run fastapi dev
```

The SD1.5 pipeline and all registry LoRA adapters load at startup (FLUX is loaded lazily on
its first request). CORS currently allows all origins (`allow_origins=["*"]` in
`app/main.py`) — tighten this before any non-demo deployment.
