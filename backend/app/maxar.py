"""Maxar Open Data handler.

Maxar's Open Data programme is a *static* STAC catalogue (no /search), organised per
disaster event. We use the community opengeos/maxar-open-data per-event GeoJSON indexes
for spatial filtering — each feature carries a public ``visual`` COG URL (~0.5 m), a
``datetime``, ``tile:clouds_percent`` and a Polygon geometry.

Discovery:
  * list_events()  -> events from the opengeos `datasets/` directory listing.
  * search_event() -> fetch one event's GeoJSON, filter by bbox / cloud / date.

Everything is cached in-process; URLs are public S3 so no signing is needed.
"""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from typing import Any

from shapely.geometry import box, shape

GITHUB_DATASETS_API = (
    "https://api.github.com/repos/opengeos/maxar-open-data/contents/datasets"
)
GEOJSON_BASE = (
    "https://raw.githubusercontent.com/opengeos/maxar-open-data/master/datasets"
)

BBox = tuple[float, float, float, float]

# A bbox covering ~the whole world means "no spatial filter" (initial unzoomed view).
_WORLD = (-179.9, -85.0, 179.9, 85.0)


def _fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "satellite-genfill"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


@lru_cache(maxsize=1)
def list_events() -> list[dict]:
    """Available Maxar events, from the opengeos `datasets/` listing."""
    entries = _fetch_json(GITHUB_DATASETS_API)
    events: list[dict] = []
    for entry in entries:
        name = entry.get("name", "")
        if not name.endswith(".geojson") or name.endswith("_union.geojson"):
            continue
        event_id = name[: -len(".geojson")]
        events.append({"id": event_id, "label": _prettify(event_id)})
    events.sort(key=lambda e: e["label"])
    return events


@lru_cache(maxsize=16)
def _event_features(event: str) -> tuple[dict, ...]:
    data = _fetch_json(f"{GEOJSON_BASE}/{event}.geojson")
    return tuple(data.get("features", []))


def search_event(
    event: str,
    bbox: BBox,
    max_cloud_cover: float | None,
    limit: int,
    datetime: str | None = None,
) -> list[dict]:
    features = _event_features(event)

    # Treat a near-world bbox as "no spatial filter" (e.g. unzoomed initial view).
    spatial = box(*bbox) if not _is_world(bbox) else None
    start, end = _parse_interval(datetime)

    scenes: list[dict] = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry")
        if geom is None:
            continue

        cloud = props.get("tile:clouds_percent")
        if (
            max_cloud_cover is not None
            and cloud is not None
            and cloud > max_cloud_cover
        ):
            continue

        dt = props.get("datetime")
        if start and dt and dt < start:
            continue
        if end and dt and dt > end:
            continue

        visual = props.get("visual")
        if not visual:
            continue

        shp = shape(geom)
        if spatial is not None and not shp.intersects(spatial):
            continue

        minx, miny, maxx, maxy = shp.bounds
        scenes.append(
            {
                "id": props.get("grid:code") or props.get("quadkey") or visual,
                "datetime": dt,
                "cloud_cover": cloud,
                "collection": event,
                "bbox": [minx, miny, maxx, maxy],
                "visual_href": visual,
                "thumbnail": None,
            }
        )

    scenes.sort(key=lambda s: s["datetime"] or "", reverse=True)
    return scenes[:limit]


def _is_world(bbox: BBox) -> bool:
    return (
        bbox[0] <= _WORLD[0]
        and bbox[1] <= _WORLD[1]
        and bbox[2] >= _WORLD[2]
        and bbox[3] >= _WORLD[3]
    )


def _parse_interval(datetime: str | None) -> tuple[str | None, str | None]:
    if not datetime or "/" not in datetime:
        return None, None
    start, end = datetime.split("/", 1)
    return (start or None), (end or None)


def _prettify(event_id: str) -> str:
    return event_id.replace("-", " ").replace("_", " ").strip()
