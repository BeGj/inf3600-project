"""Cloud Optimized GeoTIFF I/O for the inpaint endpoint.

Two responsibilities, both keyed to the same WGS-84 bbox so the read patch and the
mask are pixel-aligned at the model resolution:

  read_patch(href, bbox)      -> 512x512 RGB PIL image of the COG window
  rasterize_mask(geojson, bbox) -> 512x512 binary PIL mask (white = repaint)

The frontend works in EPSG:4326 (lon/lat). Sentinel-2 COGs are typically in a UTM
CRS, so we reproject the bbox into the source CRS before windowed reading.
"""

from __future__ import annotations

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import from_bounds
from shapely.geometry import shape

RESOLUTION = 512
WGS84 = "EPSG:4326"

# rasterio reads remote COGs over HTTP; these env tweaks keep range requests sane.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MULTIRANGE": "YES",
}

BBox = tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max


def pad_bbox(bbox: BBox, margin: float) -> BBox:
    """Grow a bbox outward by `margin` of its size on each side.

    margin=0.5 doubles the span in each dimension (the original bbox ends up centered
    and occupying the middle ~50% of the width/height). Used to give the inpainting
    model real surrounding imagery to condition on instead of regenerating the whole frame.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    dlon = (lon_max - lon_min) * margin
    dlat = (lat_max - lat_min) * margin
    return (lon_min - dlon, lat_min - dlat, lon_max + dlon, lat_max + dlat)


def read_patch(href: str, bbox: BBox) -> Image.Image:
    """Read the COG window covering `bbox` (WGS-84) and return a 512x512 RGB image."""
    lon_min, lat_min, lon_max, lat_max = bbox
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            # Reproject the WGS-84 bbox into the COG's native CRS.
            left, bottom, right, top = transform_bounds(
                WGS84, src.crs, lon_min, lat_min, lon_max, lat_max
            )
            window = from_bounds(left, bottom, right, top, transform=src.transform)
            band_count = min(src.count, 3)
            data = src.read(
                indexes=list(range(1, band_count + 1)),
                window=window,
                out_shape=(band_count, RESOLUTION, RESOLUTION),
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=0,
            )

    arr = _to_uint8_rgb(data)
    return Image.fromarray(arr, mode="RGB")


def rasterize_mask(mask_geojson: dict, bbox: BBox) -> Image.Image:
    """Burn a WGS-84 polygon into a 512x512 binary mask aligned to `bbox`.

    White (255) marks the polygon interior (the region to repaint).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    # Affine mapping the 512x512 raster onto the bbox in WGS-84 (north-up).
    transform = rasterio.transform.from_bounds(
        lon_min, lat_min, lon_max, lat_max, RESOLUTION, RESOLUTION
    )
    geom = shape(mask_geojson)
    burned = rasterize(
        [(geom, 255)],
        out_shape=(RESOLUTION, RESOLUTION),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return Image.fromarray(burned, mode="L")


def _to_uint8_rgb(data: np.ndarray) -> np.ndarray:
    """Normalise a (bands, H, W) array to an (H, W, 3) uint8 RGB array."""
    bands = data.shape[0]
    if bands == 1:
        data = np.repeat(data, 3, axis=0)
    elif bands == 2:
        data = np.concatenate([data, data[:1]], axis=0)
    data = data[:3]

    if data.dtype != np.uint8:
        out = np.zeros_like(data, dtype=np.uint8)
        for i in range(3):
            band = data[i].astype(np.float32)
            lo, hi = np.percentile(band, (2, 98)) if band.size else (0.0, 1.0)
            if hi <= lo:
                hi = lo + 1.0
            out[i] = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        data = out

    return np.transpose(data, (1, 2, 0))


def reproject_geom_to_wgs84(geom: dict, src_crs: str) -> dict:
    """Helper kept for callers that need geometry reprojection (currently unused)."""
    return transform_geom(src_crs, WGS84, geom)
