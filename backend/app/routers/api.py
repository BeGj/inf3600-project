"""Public API: model listing, catalogue search, and inpainting."""

from __future__ import annotations

import base64
import io
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

from .. import catalog, geo, jobs as job_store
from ..inference import registry
from ..inference.pipeline import engine, flux_available

router = APIRouter()

BBoxArray = tuple[float, float, float, float]

# How much real imagery to read around the polygon, as a fraction of the polygon's
# bbox size per side. 0.5 doubles the read extent, leaving the polygon in the middle
# ~50% so the inpainting model can blend with the surrounding context. Override with
# the INPAINT_CONTEXT_MARGIN env var.
CONTEXT_MARGIN = float(os.environ.get("INPAINT_CONTEXT_MARGIN", "0.5"))


# ---- /models ---------------------------------------------------------------


def _model_availability(entry: registry.ModelEntry) -> tuple[bool, str | None]:
    """Whether `entry` can run on this backend. sd15 is always available; flux-fill
    depends on GPU/VRAM/credentials (see pipeline.flux_available)."""
    if entry.family == "flux-fill":
        return flux_available()
    return True, None


@router.get("/models")
def get_models() -> list[dict]:
    out: list[dict] = []
    for m in registry.list_models():
        available, reason = _model_availability(m)
        out.append(m.public(available=available, disabled_reason=reason))
    return out


# ---- catalogues ------------------------------------------------------------


@router.get("/catalogs")
def get_catalogs() -> list[dict]:
    return catalog.list_catalogs()


@router.get("/catalog/events")
def get_catalog_events(catalog_id: str = Query("maxar", alias="catalog")) -> list[dict]:
    try:
        return catalog.list_events(catalog_id)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown catalog: {catalog_id}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to list events: {exc}") from exc


class CatalogSearchRequest(BaseModel):
    bbox: BBoxArray
    catalog: str = catalog.DEFAULT_CATALOG
    event: str | None = None
    datetime: str | None = None
    limit: int = Field(default=12, ge=1, le=50)
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)


@router.post("/catalog/search")
def catalog_search(req: CatalogSearchRequest) -> list[dict]:
    try:
        return catalog.search(
            catalog_id=req.catalog,
            bbox=req.bbox,
            datetime=req.datetime,
            event=req.event,
            limit=req.limit,
            max_cloud_cover=req.max_cloud_cover,
        )
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown catalog: {req.catalog}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # surface STAC / network errors as 502
        raise HTTPException(status_code=502, detail=f"Catalogue search failed: {exc}") from exc


# ---- /inpaint --------------------------------------------------------------


class InpaintRequest(BaseModel):
    image_url: str
    bbox: BBoxArray  # WGS-84 extent that the mask polygon spans
    mask_geojson: dict  # GeoJSON Polygon in WGS-84
    prompt: str
    model_id: str
    seed: int | None = None
    negative_prompt: str | None = None
    guidance_scale: float | None = Field(default=None, ge=0, le=30)
    strength: float | None = Field(default=None, ge=0, le=1)
    num_inference_steps: int | None = Field(default=None, ge=1, le=150)


class InpaintResponse(BaseModel):
    image_b64: str
    bbox: BBoxArray


@router.post("/inpaint")
def inpaint(req: InpaintRequest):
    try:
        entry = registry.get_model(req.model_id)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {req.model_id}")

    # Defensive: the frontend already greys out unavailable models, but a direct API call
    # could still request one (e.g. FLUX without the hardware).
    available, reason = _model_availability(entry)
    if not available:
        raise HTTPException(status_code=503, detail=reason or "Model unavailable on this backend.")

    # Read a larger area than the polygon so the inpainting model has real surrounding
    # imagery to condition on (otherwise it just generates fresh content with no context).
    # Both the patch and the mask are computed over this padded extent, and the result is
    # overlaid at the same padded extent — the polygon stays in its true location.
    context_bbox = geo.pad_bbox(req.bbox, CONTEXT_MARGIN)
    try:
        image = geo.read_patch(req.image_url, context_bbox)
        mask = geo.rasterize_mask(req.mask_geojson, context_bbox)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read imagery/mask: {exc}") from exc

    if entry.family != "flux-fill":
        # SD1.5: synchronous — return the result directly, same behavior as before.
        try:
            result = engine.infer(
                entry,
                image,
                mask,
                req.prompt,
                seed=req.seed,
                negative_prompt=req.negative_prompt,
                guidance_scale=req.guidance_scale,
                strength=req.strength,
                num_inference_steps=req.num_inference_steps,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        # The model returns a full square covering the bbox, but only the polygon was
        # repainted. Use the polygon mask as the alpha channel so the overlay shows the
        # generated content inside the polygon and stays transparent everywhere else —
        # the live map then shows through the bbox corners instead of a black/regenerated
        # rectangle.
        rgba = _apply_mask_alpha(result, mask)
        buf = io.BytesIO()
        rgba.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return InpaintResponse(image_b64=image_b64, bbox=context_bbox)

    # FLUX: downloading the model (34 GB) and running inference both block for a very long
    # time. Return a job ID immediately (202 Accepted) and let the client poll /jobs/{id}.
    job = job_store.create_job()
    needs_download = engine._flux_pipe is None
    job.set_status(
        "downloading_model" if needs_download else "queued",
        "Downloading FLUX model (~34 GB)… This only happens once." if needs_download else "Queued",
    )

    def _run() -> None:
        try:
            if engine._flux_pipe is None:
                job.set_status(
                    "downloading_model",
                    "Downloading FLUX model (~34 GB)… This only happens once.",
                )

            def _on_loaded() -> None:
                job.set_status("running", "Model ready, running inference…")

            result = engine.infer(
                entry,
                image,
                mask,
                req.prompt,
                seed=req.seed,
                negative_prompt=req.negative_prompt,
                guidance_scale=req.guidance_scale,
                strength=req.strength,
                num_inference_steps=req.num_inference_steps,
                on_flux_loaded=_on_loaded,
            )
            rgba = _apply_mask_alpha(result, mask)
            buf = io.BytesIO()
            rgba.save(buf, format="PNG")
            job.result_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            job.result_bbox = list(context_bbox)
            job.set_status("done")
        except Exception as exc:
            job.error = str(exc)
            job.set_status("error", str(exc))

    job_store.submit(_run)
    return JSONResponse(status_code=202, content={"job_id": job.id})


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_response()


def _apply_mask_alpha(result: Image.Image, mask: Image.Image) -> Image.Image:
    """Return `result` as RGBA with `mask` (white = keep) as the alpha channel.

    The mask is resized to the result, softly feathered so the inpainted patch blends
    into the surrounding map rather than showing a hard rectangular/polygon seam.
    """
    rgba = result.convert("RGBA")
    alpha = mask.convert("L").resize(result.size, Image.Resampling.NEAREST)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=2))
    rgba.putalpha(alpha)
    return rgba
