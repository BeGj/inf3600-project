import type { Polygon } from "geojson";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface BBox {
  lngMin: number;
  latMin: number;
  lngMax: number;
  latMax: number;
}

export interface InpaintResult {
  image_b64: string;
  bbox: [number, number, number, number];
}

function bboxToArray(b: BBox): [number, number, number, number] {
  return [b.lngMin, b.latMin, b.lngMax, b.latMax];
}

export async function inpaint(
  bbox: BBox,
  maskGeojson: Polygon,
  prompt: string,
  imageUrl: string
): Promise<InpaintResult> {
  const res = await fetch(`${BASE_URL}/inpaint`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      bbox: bboxToArray(bbox),
      mask_geojson: maskGeojson,
      prompt,
      image_url: imageUrl,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Inpaint failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function cloudRemove(
  bbox: BBox,
  imageUrl: string
): Promise<InpaintResult> {
  const res = await fetch(`${BASE_URL}/cloud-remove`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      bbox: bboxToArray(bbox),
      image_url: imageUrl,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Cloud remove failed (${res.status}): ${detail}`);
  }
  return res.json();
}
