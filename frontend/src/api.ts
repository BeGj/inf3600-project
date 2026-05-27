import type { Polygon } from "geojson";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface BBox {
  lngMin: number;
  latMin: number;
  lngMax: number;
  latMax: number;
}

export interface ModelInfo {
  id: string;
  label: string;
  type: "base" | "lora";
  family: string; // "sd15" | "flux-fill" — drives which inference params apply
  default_prompt: string;
  negative_prompt: string;
  available: boolean; // false when backend hardware can't run it (e.g. FLUX, no big GPU)
  disabled_reason: string | null;
}

export interface Scene {
  id: string;
  datetime: string | null;
  cloud_cover: number | null;
  collection: string;
  bbox: [number, number, number, number];
  visual_href: string;
  thumbnail: string | null;
}

export interface InpaintResult {
  image_b64: string;
  bbox: [number, number, number, number];
}

export interface CatalogInfo {
  id: string;
  label: string;
  kind: string;
  resolution_m: number;
  coverage: string;
  supports_cloud: boolean;
  supports_datetime: boolean;
  requires_event: boolean;
}

export interface CatalogEvent {
  id: string;
  label: string;
}

export interface CatalogSearchParams {
  bbox: BBox;
  catalog: string;
  event?: string;
  datetime?: string;
  maxCloudCover?: number;
  limit?: number;
}

function bboxToArray(b: BBox): [number, number, number, number] {
  return [b.lngMin, b.latMin, b.lngMax, b.latMax];
}

export async function getModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${BASE_URL}/models`);
  if (!res.ok) throw new Error(`Failed to load models (${res.status})`);
  return res.json();
}

export async function getCatalogs(): Promise<CatalogInfo[]> {
  const res = await fetch(`${BASE_URL}/catalogs`);
  if (!res.ok) throw new Error(`Failed to load catalogues (${res.status})`);
  return res.json();
}

export async function getEvents(catalogId: string): Promise<CatalogEvent[]> {
  const res = await fetch(`${BASE_URL}/catalog/events?catalog=${encodeURIComponent(catalogId)}`);
  if (!res.ok) throw new Error(`Failed to load events (${res.status})`);
  return res.json();
}

export async function searchCatalog(params: CatalogSearchParams): Promise<Scene[]> {
  const res = await fetch(`${BASE_URL}/catalog/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      bbox: bboxToArray(params.bbox),
      catalog: params.catalog,
      event: params.event,
      datetime: params.datetime,
      max_cloud_cover: params.maxCloudCover,
      limit: params.limit,
    }),
  });
  if (!res.ok) {
    throw new Error(`Catalogue search failed (${res.status}): ${await res.text()}`);
  }
  return res.json();
}

export interface InpaintOptions {
  negativePrompt?: string;
  guidanceScale?: number;
  strength?: number;
  numInferenceSteps?: number;
}

export async function inpaint(
  bbox: BBox,
  maskGeojson: Polygon,
  prompt: string,
  imageUrl: string,
  modelId: string,
  opts: InpaintOptions = {}
): Promise<InpaintResult> {
  const body: Record<string, unknown> = {
    bbox: bboxToArray(bbox),
    mask_geojson: maskGeojson,
    prompt,
    image_url: imageUrl,
    model_id: modelId,
  };
  if (opts.negativePrompt !== undefined) body.negative_prompt = opts.negativePrompt;
  if (opts.guidanceScale !== undefined) body.guidance_scale = opts.guidanceScale;
  if (opts.strength !== undefined) body.strength = opts.strength;
  if (opts.numInferenceSteps !== undefined) body.num_inference_steps = opts.numInferenceSteps;

  const res = await fetch(`${BASE_URL}/inpaint`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Inpaint failed (${res.status}): ${detail}`);
  }
  return res.json();
}
