"""Prepare a Norwegian / North European satellite imagery dataset.

For each land-cover class, samples geographic tiles near known Nordic anchor locations,
reads the ESA WorldCover 2021 v200 land-cover raster (free 10m COG on S3) to confirm
class coverage and generate the inpaint mask, downloads the best cloud-free Sentinel-2
patch from Earth Search, and saves image+mask pairs to datasets/osm_nordic/ in the same
format as prepare_houses.py and prepare_trees.py.

ESA WorldCover replaces the Overpass API (which is blocked in this environment).
WorldCover is pixel-aligned with Sentinel-2 at 10m/px, requires no API key, and
provides the land-cover classes we need directly as a COG.

WorldCover class values used:
  10 → tree cover   (boreal_forest)
  20 → shrubland    (heath)
  30 → grassland    (grassland)
  40 → cropland     (farmland)
  50 → built-up     (nordic_urban, industrial, airport — distinguished by anchor)
  60 → bare/sparse  (alpine)
  80 → water        (water)

Usage:
    cd notebooks
    uv run python prepare/prepare_osm_sentinel.py --output-dir ../datasets/osm_nordic

Quick test (3 tiles per class):
    uv run python prepare/prepare_osm_sentinel.py \\
        --output-dir ../datasets/nordic_test --per-class 3 --limit 27
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform
from PIL import Image
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
# ESA WorldCover 2021 v200 — public COG on S3, 3°×3° tiles
WORLDCOVER_URL_TEMPLATE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
RESOLUTION = 512
WGS84 = "EPSG:4326"

BBox = tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MULTIRANGE": "YES",
}

# ─── Land-cover class definitions ────────────────────────────────────────────


@dataclass
class ClassDef:
    name: str
    wc_values: list[int]  # ESA WorldCover pixel values that count as this class
    prompt: str
    anchors: list[tuple[float, float]]  # [(lon, lat), ...]
    # Per-class overrides; None falls back to the CLI --jitter / --min-coverage values
    jitter: float | None = None
    min_coverage: float | None = None
    max_coverage: float = (
        0.90  # tiles above this have too little context for inpainting
    )


LAND_COVER_CLASSES: dict[str, ClassDef] = {
    "boreal_forest": ClassDef(
        name="boreal_forest",
        wc_values=[10],  # Tree cover
        prompt="satellite view of Scandinavian boreal forest, dense dark coniferous canopy",
        anchors=[
            (11.5, 61.5),  # Østerdalen, Norway
            (12.0, 62.3),  # Femundsmarka, Norway
            (14.5, 61.0),  # Dalarna, Sweden
            (27.0, 62.5),  # Finnish Lakeland
            (10.5, 63.5),  # Trøndelag, Norway
        ],
        jitter=1.5,
        min_coverage=0.30,
    ),
    "nordic_urban": ClassDef(
        name="nordic_urban",
        wc_values=[50],  # Built-up
        prompt="satellite view of Norwegian residential neighborhood, timber houses, streets",
        anchors=[
            (10.75, 59.91),  # Oslo city centre
            (5.32, 60.39),  # Bergen
            (10.40, 63.43),  # Trondheim
            (5.73, 58.97),  # Stavanger
            (18.07, 59.33),  # Stockholm
            (24.94, 60.17),  # Helsinki
            (10.55, 59.68),  # Ski / Follo, Oslo suburb
            (10.40, 59.77),  # Asker, Oslo suburb
        ],
        # Tight jitter keeps tiles over city centres; urban fabric rarely exceeds 40%
        # in a 5 km tile even in dense Nordic cities.
        jitter=0.08,
        min_coverage=0.15,
    ),
    "farmland": ClassDef(
        name="farmland",
        wc_values=[40],  # Cropland
        prompt="satellite view of Nordic agricultural fields, patchwork cropland",
        anchors=[
            (5.60, 58.75),  # Jæren, Norway — flattest farmland in Norway
            (10.30, 59.20),  # Vestfold, Norway
            (13.50, 55.70),  # Skåne, Sweden
            (9.50, 56.00),  # Jutland, Denmark
            (15.50, 58.40),  # Östergötland, Sweden
            (11.20, 59.10),  # Østfold, Norway
            (10.20, 55.90),  # Funen, Denmark
        ],
        jitter=0.4,
        min_coverage=0.25,
    ),
    "water": ClassDef(
        name="water",
        wc_values=[80],  # Permanent water bodies
        prompt="satellite view of Norwegian fjord or Scandinavian lake, dark cold water",
        anchors=[
            (6.30, 61.00),  # Sognefjord
            (6.50, 60.30),  # Hardangerfjord
            (10.50, 59.40),  # Oslofjord
            (10.80, 60.70),  # Mjøsa lake, Norway
            (11.80, 62.00),  # Femunden lake, Norway
            (13.00, 58.90),  # Vänern, Sweden
            (28.20, 61.30),  # Saimaa, Finland
        ],
        # Lower min so shore tiles (fjord edge + mountain) are included.
        # Lower max so fully-open-water tiles without any land context are rejected.
        jitter=0.6,
        min_coverage=0.20,
        max_coverage=0.80,
    ),
    "alpine": ClassDef(
        name="alpine",
        wc_values=[60, 100],  # Bare/sparse vegetation + moss/lichen
        prompt="satellite view of Norwegian alpine terrain, rocky mountain above treeline",
        anchors=[
            (8.4, 61.6),  # Jotunheimen
            (7.5, 60.4),  # Hardangervidda
            (9.8, 62.0),  # Rondane
            (9.5, 62.3),  # Dovrefjell
            (9.0, 62.8),  # Trollheimen
        ],
        jitter=1.0,
        min_coverage=0.25,
    ),
    "heath": ClassDef(
        name="heath",
        wc_values=[20],  # Shrubland — Norwegian coastal heathland, Scottish-style moor
        prompt="satellite view of Scandinavian heathland, open low vegetation, moorland",
        anchors=[
            (5.50, 58.50),  # Jæren coast, Norway
            (7.00, 58.20),  # Vest-Agder coast, Norway
            (8.50, 58.10),  # Aust-Agder, Norway
            (5.30, 59.50),  # Hordaland coast, Norway
            (9.00, 57.00),  # Jutland heathland, Denmark
            (9.10, 56.00),  # Kongenshus heath reserve, Denmark
            (8.60, 56.30),  # Harrild heath, Denmark
            (-3.00, 57.50),  # Scottish Highlands — extensive shrubland/heath
            (-4.50, 58.00),  # Caithness moor, Scotland
        ],
        # WorldCover shrubland is patchy in Norway; lower threshold catches mixed tiles
        jitter=0.5,
        min_coverage=0.10,
    ),
    "industrial": ClassDef(
        name="industrial",
        wc_values=[50],  # Built-up (at industrial/port anchors)
        prompt="satellite view of Scandinavian industrial zone, warehouses, large structures",
        anchors=[
            (10.50, 59.75),  # Oslo Alnabru industrial
            (5.02, 60.81),  # Mongstad refinery, Norway
            (17.40, 68.40),  # Narvik industrial port
            (11.90, 57.70),  # Gothenburg port, Sweden
            (24.50, 65.70),  # Kemi, Finland
            (10.97, 59.91),  # Romerike logistics, Norway
        ],
        jitter=0.04,
        min_coverage=0.15,
    ),
    "grassland": ClassDef(
        name="grassland",
        wc_values=[30],  # Grassland
        prompt="satellite view of Scandinavian open grassland, meadow, green pasture",
        anchors=[
            (10.60, 59.50),  # Akershus, Norway
            (11.00, 60.50),  # Hedmark, Norway
            (14.00, 56.50),  # Småland, Sweden
            (10.00, 56.20),  # Fyn, Denmark
            (25.00, 60.50),  # Southern Finland
            (9.20, 56.50),  # Jutland grassland, Denmark
            (12.50, 56.00),  # Skåne coast grassland, Sweden
        ],
        jitter=0.6,
        min_coverage=0.15,
    ),
    "airport": ClassDef(
        name="airport",
        wc_values=[50],  # Built-up (at airport anchors — very tight jitter)
        prompt="satellite view of Norwegian airport, runways, taxiways, terminal buildings",
        anchors=[
            (11.10, 60.20),  # Oslo Gardermoen
            (5.23, 60.29),  # Bergen Flesland
            (5.63, 58.88),  # Stavanger Sola
            (10.93, 63.46),  # Trondheim Værnes
            (17.93, 59.65),  # Stockholm Arlanda
            (24.96, 60.33),  # Helsinki-Vantaa
        ],
        # Nearly zero jitter: tiles must stay over the airport footprint
        jitter=0.02,
        min_coverage=0.20,
    ),
}

# ─── Tile candidate generation ────────────────────────────────────────────────


def generate_tile_candidates(
    class_def: ClassDef,
    n: int,
    tile_deg: float,
    default_jitter: float,
    seed: int,
) -> list[BBox]:
    """Generate n random bbox candidates by jittering anchor centroids."""
    jitter = class_def.jitter if class_def.jitter is not None else default_jitter
    rng = random.Random(seed)
    half = tile_deg / 2.0
    candidates: list[BBox] = []
    for _ in range(n):
        lon_c, lat_c = rng.choice(class_def.anchors)
        lon = lon_c + rng.uniform(-jitter, jitter)
        lat = lat_c + rng.uniform(-jitter, jitter)
        lat = max(-85.0, min(85.0, lat))
        candidates.append((lon - half, lat - half, lon + half, lat + half))
    return candidates


# ─── ESA WorldCover ───────────────────────────────────────────────────────────


def _worldcover_tile_name(lat: float, lon: float) -> str:
    """Return the WorldCover 3°×3° tile name covering the given point."""
    lat_floor = math.floor(lat / 3) * 3
    lon_floor = math.floor(lon / 3) * 3
    lat_str = f"N{abs(lat_floor):02d}" if lat_floor >= 0 else f"S{abs(lat_floor):02d}"
    lon_str = f"E{abs(lon_floor):03d}" if lon_floor >= 0 else f"W{abs(lon_floor):03d}"
    return f"{lat_str}{lon_str}"


def worldcover_tile_url(lat: float, lon: float) -> str:
    return WORLDCOVER_URL_TEMPLATE.format(tile=_worldcover_tile_name(lat, lon))


def read_worldcover_patch(bbox: BBox) -> np.ndarray | None:
    """Read ESA WorldCover class values for bbox at RESOLUTION×RESOLUTION pixels.

    Returns uint8 array of shape (RESOLUTION, RESOLUTION) with WorldCover class IDs,
    or None on failure.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_center = (lat_min + lat_max) / 2.0
    lon_center = (lon_min + lon_max) / 2.0
    url = worldcover_tile_url(lat_center, lon_center)
    try:
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(url) as src:
                left, bottom, right, top = transform_bounds(
                    WGS84, src.crs, lon_min, lat_min, lon_max, lat_max
                )
                window = from_bounds(left, bottom, right, top, transform=src.transform)
                data = src.read(
                    indexes=[1],
                    window=window,
                    out_shape=(1, RESOLUTION, RESOLUTION),
                    resampling=Resampling.nearest,  # categorical data: no interpolation
                    boundless=True,
                    fill_value=0,
                )
        return data[0].astype(np.uint8)
    except Exception:
        return None


def wc_coverage(lc_patch: np.ndarray, target_values: list[int]) -> float:
    """Fraction of pixels in lc_patch that belong to any of target_values."""
    mask = np.zeros(lc_patch.shape, dtype=bool)
    for v in target_values:
        mask |= lc_patch == v
    return float(mask.mean())


def wc_mask_image(lc_patch: np.ndarray, target_values: list[int]) -> Image.Image:
    """Binary mask: white (255) = target class pixels, black (0) = everything else."""
    mask = np.zeros(lc_patch.shape, dtype=bool)
    for v in target_values:
        mask |= lc_patch == v
    return Image.fromarray(np.where(mask, np.uint8(255), np.uint8(0)), mode="L")


# ─── Sentinel-2 scene lookup ─────────────────────────────────────────────────

_stac_client: Client | None = None


def _get_stac_client() -> Client:
    global _stac_client
    if _stac_client is None:
        _stac_client = Client.open(EARTH_SEARCH_URL)
    return _stac_client


def fetch_sentinel_href(
    bbox: BBox,
    date_range: str,
    max_cloud: float,
) -> tuple[str, float, str, bool] | None:
    """Return (visual_href, cloud_cover, datetime_iso, is_cloud_fallback) or None."""
    client = _get_stac_client()
    for cloud_limit, is_fallback in [(max_cloud, False), (max_cloud + 10.0, True)]:
        try:
            results = client.search(
                collections=["sentinel-2-l2a"],
                bbox=list(bbox),
                datetime=date_range,
                query={"eo:cloud_cover": {"lte": cloud_limit}},
                max_items=5,
                sortby=[{"field": "properties.datetime", "direction": "desc"}],
            )
            for item in results.items():
                visual = item.assets.get("visual")
                if visual is None:
                    continue
                return (
                    visual.href,
                    item.properties.get("eo:cloud_cover", 0.0),
                    item.datetime.isoformat() if item.datetime else None,
                    is_fallback,
                )
        except Exception:
            continue
    return None


# ─── COG patch reading (inline copy from backend/app/geo.py) ─────────────────


def _to_uint8_rgb(data: np.ndarray) -> np.ndarray:
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


def read_patch_inline(href: str, bbox: BBox) -> Image.Image | None:
    lon_min, lat_min, lon_max, lat_max = bbox
    try:
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(href) as src:
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
    except Exception:
        return None
    # Reject swath-edge / missing-data tiles: pixels where all bands are exactly 0
    # are rasterio fill values, not real dark pixels.  More than 1% fill → data gap.
    if data.shape[0] >= 2 and np.all(data == 0, axis=0).mean() > 0.01:
        return None
    return Image.fromarray(_to_uint8_rgb(data), mode="RGB")


# ─── Quality guards ───────────────────────────────────────────────────────────


def is_too_bright(
    image: Image.Image, threshold: int = 210, max_fraction: float = 0.25
) -> bool:
    """Detect snow- or cloud-contaminated patches by near-white pixel fraction."""
    arr = np.array(image)
    bright = (
        (arr[:, :, 0] > threshold)
        & (arr[:, :, 1] > threshold)
        & (arr[:, :, 2] > threshold)
    )
    return float(bright.mean()) > max_fraction


# ─── Per-tile pipeline ────────────────────────────────────────────────────────


def process_tile(
    idx: int,
    bbox: BBox,
    class_def: ClassDef,
    images_dir: Path,
    masks_dir: Path,
    date_range: str,
    max_cloud: float,
    default_min_coverage: float,
    verbose: bool = False,
) -> dict | None:
    def reject(reason: str) -> None:
        if verbose:
            print(f"    skip: {reason}")

    min_cov = (
        class_def.min_coverage
        if class_def.min_coverage is not None
        else default_min_coverage
    )
    max_cov = class_def.max_coverage

    # 1. Read WorldCover land-cover patch
    lc_patch = read_worldcover_patch(bbox)
    if lc_patch is None:
        reject("WorldCover read failed")
        return None

    # 2. Check class coverage
    coverage = wc_coverage(lc_patch, class_def.wc_values)
    if coverage < min_cov:
        reject(f"WorldCover coverage {coverage:.1%} < min {min_cov:.0%}")
        return None
    if coverage > max_cov:
        reject(
            f"WorldCover coverage {coverage:.1%} > {max_cov:.0%} (too little context)"
        )
        return None

    # 3. Fetch best Sentinel-2 scene
    scene = fetch_sentinel_href(bbox, date_range, max_cloud)
    if scene is None:
        reject("no Sentinel-2 scene found")
        return None
    href, cloud_cover, scene_datetime, cloud_fallback = scene

    # 4. Download COG patch
    image = read_patch_inline(href, bbox)
    if image is None:
        reject("COG download failed or >1% fill pixels (swath edge / data gap)")
        return None

    # 5. Snow / cloud guard
    if is_too_bright(image):
        reject("image too bright (snow or cloud)")
        return None

    # 6. Generate mask from WorldCover (same pixels, binary)
    mask_img = wc_mask_image(lc_patch, class_def.wc_values)

    # 7. Save to disk
    stem = f"{idx:06d}"
    image_name = f"{stem}.png"
    mask_name = f"{stem}_mask.png"
    image.save(images_dir / image_name)
    mask_img.save(masks_dir / mask_name)

    return {
        "image": f"images/{image_name}",
        "mask": f"masks/{mask_name}",
        "prompt": class_def.prompt,
        "land_cover_class": class_def.name,
        "bbox": list(bbox),
        "scene_datetime": scene_datetime,
        "cloud_cover": cloud_cover,
        "cloud_fallback": cloud_fallback,
        "mask_coverage": round(coverage, 4),
        "source": "sentinel-2+worldcover",
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Norwegian/North European Sentinel-2 + ESA WorldCover inpainting dataset."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "datasets" / "osm_nordic"
    )
    parser.add_argument(
        "--per-class", type=int, default=150, help="Target tiles per class"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.30,
        help="Min WorldCover class fraction per tile (default 0.30)",
    )
    parser.add_argument(
        "--max-cloud", type=float, default=15.0, help="Max Sentinel-2 cloud cover %%"
    )
    parser.add_argument("--date-range", type=str, default="2023-05-01/2024-09-30")
    parser.add_argument(
        "--tile-deg",
        type=float,
        default=0.046,
        help="Tile width/height in degrees (default 0.046 ≈ 5 km at S2 10m/px)",
    )
    parser.add_argument(
        "--jitter", type=float, default=1.5, help="Anchor jitter radius in degrees"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=0, help="Hard cap on total tiles (0 = unlimited)"
    )
    parser.add_argument(
        "--classes", nargs="*", default=None, help="Subset of class names to process"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print rejection reason per discarded tile",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    classes = (
        {k: v for k, v in LAND_COVER_CLASSES.items() if k in args.classes}
        if args.classes
        else LAND_COVER_CLASSES
    )

    all_records: list[dict] = []
    global_idx = 0
    per_class_counts: dict[str, int] = {}

    for class_name, class_def in classes.items():
        print(f"\n--- {class_name} (WorldCover values: {class_def.wc_values}) ---")
        count = 0
        candidates = generate_tile_candidates(
            class_def,
            n=args.per_class * 15,
            tile_deg=args.tile_deg,
            default_jitter=args.jitter,
            seed=args.seed,
        )

        pbar = tqdm(candidates, desc=class_name, unit="tile")
        for bbox in pbar:
            if count >= args.per_class:
                break
            if args.limit > 0 and global_idx >= args.limit:
                break

            record = process_tile(
                idx=global_idx,
                bbox=bbox,
                class_def=class_def,
                images_dir=images_dir,
                masks_dir=masks_dir,
                date_range=args.date_range,
                max_cloud=args.max_cloud,
                default_min_coverage=args.min_coverage,
                verbose=args.verbose,
            )

            if record is not None:
                all_records.append(record)
                count += 1
                global_idx += 1

            pbar.set_postfix({"saved": count, "tried": pbar.n})

        per_class_counts[class_name] = count
        print(f"  saved {count}/{args.per_class} tiles")

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record) + "\n")

    summary = {
        "source": "sentinel-2+worldcover",
        "region": "Norwegian/North European",
        "per_class_target": args.per_class,
        "count": len(all_records),
        "per_class_counts": per_class_counts,
        "output_dir": str(output_dir.resolve()),
        "tile_deg": args.tile_deg,
        "min_coverage": args.min_coverage,
        "max_cloud": args.max_cloud,
        "date_range": args.date_range,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
