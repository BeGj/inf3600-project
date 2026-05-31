"""Imagery catalogue registry + search dispatch.

Two kinds of source sit behind one interface:

  * ``stac-api``       — a searchable STAC API queried with pystac-client
                         (Sentinel-2 via Earth Search: global, 10 m).
  * ``maxar-opendata`` — Maxar's *static* Open Data catalogue, organised per disaster
                         event, queried via the opengeos per-event GeoJSON index
                         (see app/maxar.py). ~0.5 m, public COGs, disaster areas only.

All sources return the same scene dict shape so the frontend and /inpaint are agnostic:
    {id, datetime, cloud_cover, collection, bbox, visual_href, thumbnail}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from pystac_client import Client

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_LIMIT = 50

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class CatalogDef:
    id: str
    label: str
    kind: str  # "stac-api" | "maxar-opendata"
    resolution_m: float
    coverage: str
    supports_cloud: bool = True
    supports_datetime: bool = True
    requires_event: bool = False
    # stac-api specifics:
    stac_url: str | None = None
    collections: list[str] = field(default_factory=list)
    asset_key: str = "visual"

    def public(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "resolution_m": self.resolution_m,
            "coverage": self.coverage,
            "supports_cloud": self.supports_cloud,
            "supports_datetime": self.supports_datetime,
            "requires_event": self.requires_event,
        }


CATALOGS: dict[str, CatalogDef] = {
    "sentinel-2": CatalogDef(
        id="sentinel-2",
        label="Sentinel-2 (Earth Search)",
        kind="stac-api",
        resolution_m=10,
        coverage="Global",
        stac_url=EARTH_SEARCH_URL,
        collections=["sentinel-2-l2a"],
        asset_key="visual",
    ),
    "maxar": CatalogDef(
        id="maxar",
        label="Maxar Open Data (~0.5 m)",
        kind="maxar-opendata",
        resolution_m=0.5,
        coverage="Disaster events",
        supports_cloud=True,
        supports_datetime=False,
        requires_event=True,
    ),
}

DEFAULT_CATALOG = "sentinel-2"


def list_catalogs() -> list[dict]:
    return [c.public() for c in CATALOGS.values()]


def get_catalog(catalog_id: str) -> CatalogDef:
    if catalog_id not in CATALOGS:
        raise KeyError(catalog_id)
    return CATALOGS[catalog_id]


def list_events(catalog_id: str) -> list[dict]:
    cat = get_catalog(catalog_id)
    if cat.kind != "maxar-opendata":
        return []
    from . import maxar

    return maxar.list_events()


def search(
    catalog_id: str,
    bbox: BBox,
    datetime: str | None = None,
    max_cloud_cover: float | None = None,
    limit: int = DEFAULT_LIMIT,
    event: str | None = None,
) -> list[dict]:
    cat = get_catalog(catalog_id)
    if cat.kind == "stac-api":
        return _search_stac_api(cat, bbox, datetime, max_cloud_cover, limit)
    if cat.kind == "maxar-opendata":
        if not event:
            raise ValueError("Maxar search requires an 'event'.")
        from . import maxar

        return maxar.search_event(event, bbox, max_cloud_cover, limit, datetime)
    raise ValueError(f"Unknown catalogue kind: {cat.kind}")


@lru_cache(maxsize=4)
def _client(stac_url: str) -> Client:
    return Client.open(stac_url)


def _search_stac_api(
    cat: CatalogDef,
    bbox: BBox,
    datetime: str | None,
    max_cloud_cover: float | None,
    limit: int,
) -> list[dict]:
    query = None
    if max_cloud_cover is not None and cat.supports_cloud:
        query = {"eo:cloud_cover": {"lte": max_cloud_cover}}

    search_result = _client(cat.stac_url).search(
        collections=cat.collections,
        bbox=list(bbox),
        datetime=datetime,
        query=query,
        max_items=limit,
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
    )

    scenes: list[dict] = []
    for item in search_result.items():
        visual = item.assets.get(cat.asset_key)
        if visual is None:
            continue  # need the true-color COG to display + inpaint
        thumbnail = item.assets.get("thumbnail")
        scenes.append(
            {
                "id": item.id,
                "datetime": item.datetime.isoformat() if item.datetime else None,
                "cloud_cover": item.properties.get("eo:cloud_cover"),
                "collection": item.collection_id,
                "bbox": item.bbox,
                "visual_href": visual.href,
                "thumbnail": thumbnail.href if thumbnail else None,
            }
        )
    return scenes
