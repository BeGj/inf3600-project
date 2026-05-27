import { useCallback, useEffect, useRef, useState } from "react";
import type { Polygon } from "geojson";
import Map from "ol/Map";
import { transformExtent } from "ol/proj";
import MapView from "./MapView";
import type { ResultOverlay } from "./MapView";
import PromptPanel from "./PromptPanel";
import type { RestoreSnapshot } from "./PromptPanel";
import CatalogPanel from "./CatalogPanel";
import GeneratedImagesList from "./GeneratedImagesList";
import { getModels, inpaint } from "./api";
import type { BBox, InpaintOptions, InpaintStatusEvent, ModelInfo, Scene } from "./api";
import "./App.css";

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [scene, setScene] = useState<Scene | null>(null);
  const [drawingActive, setDrawingActive] = useState(false);
  const [mask, setMask] = useState<Polygon | null>(null);
  const [maskId, setMaskId] = useState<string>(() => crypto.randomUUID());
  const [preloadMask, setPreloadMask] = useState<Polygon | null>(null);
  const [overlays, setOverlays] = useState<ResultOverlay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [clearKey, setClearKey] = useState(0);
  const [hoverBBox, setHoverBBox] = useState<[number, number, number, number] | null>(null);
  const [restoreSnapshot, setRestoreSnapshot] = useState<RestoreSnapshot | null>(null);
  const mapRef = useRef<Map | null>(null);

  // Load available models once.
  useEffect(() => {
    getModels()
      .then((m) => {
        setModels(m);
        // Default to the first runnable model so a disabled one (e.g. FLUX without a
        // big-enough GPU) is never auto-selected.
        const first = m.find((x) => x.available) ?? m[0];
        if (first) setModelId(first.id);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  // Current map view extent in WGS-84, for catalogue search.
  const getViewBBox = useCallback((): BBox | null => {
    const map = mapRef.current;
    if (!map) return null;
    const extent3857 = map.getView().calculateExtent(map.getSize());
    const [lngMin, latMin, lngMax, latMax] = transformExtent(extent3857, "EPSG:3857", "EPSG:4326");
    return { lngMin, latMin, lngMax, latMax };
  }, []);

  // WGS-84 bbox spanned by the drawn mask polygon.
  const bboxFromMask = useCallback((geojson: Polygon): BBox => {
    const positions = geojson.coordinates.flat();
    const lngs = positions.map((p) => p[0]);
    const lats = positions.map((p) => p[1]);
    return { lngMin: Math.min(...lngs), latMin: Math.min(...lats), lngMax: Math.max(...lngs), latMax: Math.max(...lats) };
  }, []);

  const handleMaskDrawn = useCallback((geojson: Polygon) => {
    setMask(geojson);
    setMaskId(crypto.randomUUID());
    setPreloadMask(null);
    setDrawingActive(false);
  }, []);

  const handleInvalidDraw = useCallback((msg: string) => {
    setError(msg);
    setDrawingActive(false);
  }, []);

  const handleSelectScene = useCallback((s: Scene) => {
    setScene(s);
    setMask(null);
    setPreloadMask(null);
    setError(null);
    setClearKey((k) => k + 1);
  }, []);

  const handleHoverScene = useCallback((s: Scene | null) => {
    setHoverBBox(s ? s.bbox : null);
  }, []);

  const handleClearScene = useCallback(() => {
    setScene(null);
    setMask(null);
    setPreloadMask(null);
    setDrawingActive(false);
    setClearKey((k) => k + 1);
  }, []);

  // Fly the map to the combined extent of search results (for event catalogues whose
  // imagery is elsewhere than the user's current view).
  const fitToResults = useCallback((scenes: Scene[]) => {
    const map = mapRef.current;
    if (!map || scenes.length === 0) return;
    let [lngMin, latMin, lngMax, latMax] = scenes[0].bbox;
    for (const s of scenes) {
      lngMin = Math.min(lngMin, s.bbox[0]);
      latMin = Math.min(latMin, s.bbox[1]);
      lngMax = Math.max(lngMax, s.bbox[2]);
      latMax = Math.max(latMax, s.bbox[3]);
    }
    const viewProj = map.getView().getProjection();
    const extent = transformExtent([lngMin, latMin, lngMax, latMax], "EPSG:4326", viewProj);
    map.getView().fit(extent, { duration: 600, padding: [40, 40, 40, 40] });
  }, []);

  const handleGenerate = useCallback(
    async (prompt: string, opts: InpaintOptions) => {
      if (!scene || !mask || !modelId) return;
      setError(null);
      setStatusMessage(null);
      setLoading(true);
      const currentMaskId = maskId;
      try {
        const bbox = bboxFromMask(mask);
        const result = await inpaint(
          bbox, mask, prompt, scene.visual_href, modelId, opts,
          (evt: InpaintStatusEvent) => {
            if (evt.phase === "downloading_model") {
              const m = Math.floor(evt.elapsedTotalS / 60);
              const s = Math.floor(evt.elapsedTotalS % 60);
              const elapsed = m > 0 ? `${m}m ${s}s` : `${s}s`;
              setStatusMessage(`Downloading FLUX model (~34 GB)… This only happens once. Elapsed: ${elapsed}`);
            } else if (evt.phase === "running") {
              setStatusMessage("Running inference…");
            }
          },
        );
        const newOverlay: ResultOverlay = {
          maskId: currentMaskId,
          image_b64: result.image_b64,
          bbox: result.bbox,
          prompt,
          negativePrompt: opts.negativePrompt ?? "",
          modelId,
          mask,
          guidanceScale: opts.guidanceScale ?? 6.5,
          strength: opts.strength ?? 1.0,
          numInferenceSteps: opts.numInferenceSteps ?? 40,
        };
        setOverlays((prev) => [
          ...prev.filter((o) => o.maskId !== currentMaskId),
          newOverlay,
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
        setStatusMessage(null);
      }
    },
    [scene, mask, maskId, modelId, bboxFromMask]
  );

  const handleRemoveOverlay = useCallback((id: string) => {
    setOverlays((prev) => prev.filter((o) => o.maskId !== id));
  }, []);

  const handleZoomToOverlay = useCallback((bbox: [number, number, number, number]) => {
    const map = mapRef.current;
    if (!map) return;
    const viewProj = map.getView().getProjection();
    const extent = transformExtent(bbox, "EPSG:4326", viewProj);
    map.getView().fit(extent, { duration: 600, padding: [40, 40, 40, 40] });
  }, []);

  const handleEditOverlay = useCallback((overlay: ResultOverlay) => {
    setMask(overlay.mask);
    setMaskId(overlay.maskId);
    setPreloadMask(overlay.mask);
    setModelId(overlay.modelId);
    setRestoreSnapshot({
      key: Date.now(),
      prompt: overlay.prompt,
      negativePrompt: overlay.negativePrompt,
      guidanceScale: overlay.guidanceScale,
      strength: overlay.strength,
      numInferenceSteps: overlay.numInferenceSteps,
    });
  }, []);

  const handleDownload = useCallback(() => {
    if (overlays.length === 0) return;
    const last = overlays[overlays.length - 1];
    const a = document.createElement("a");
    a.href = `data:image/png;base64,${last.image_b64}`;
    a.download = "generated_patch.png";
    a.click();
  }, [overlays]);

  return (
    <div className="app-root">
      <div className="side-panel">
        <CatalogPanel getViewBBox={getViewBBox} selectedSceneId={scene?.id ?? null} onSelectScene={handleSelectScene} onHoverScene={handleHoverScene} onClearScene={handleClearScene} onFitResults={fitToResults} />
        <PromptPanel
          models={models}
          modelId={modelId}
          onModelChange={setModelId}
          sceneReady={!!scene}
          drawingActive={drawingActive}
          maskReady={!!mask}
          onStartDraw={() => { setMask(null); setPreloadMask(null); setDrawingActive(true); setClearKey((k) => k + 1); }}
          onClearMask={() => { setMask(null); setPreloadMask(null); setDrawingActive(false); setClearKey((k) => k + 1); }}
          onGenerate={handleGenerate}
          loading={loading}
          statusMessage={statusMessage}
          error={error}
          restoreSnapshot={restoreSnapshot}
        />
        <GeneratedImagesList
          overlays={overlays}
          onRemove={handleRemoveOverlay}
          onZoom={handleZoomToOverlay}
          onEdit={handleEditOverlay}
        />
      </div>
      <div className="map-container">
        <MapView
          cogUrl={scene?.visual_href ?? null}
          overlays={overlays}
          drawingActive={drawingActive}
          clearKey={clearKey}
          highlightBBox={hoverBBox}
          preloadMask={preloadMask}
          onMaskDrawn={handleMaskDrawn}
          onInvalidDraw={handleInvalidDraw}
          onMapReady={(map) => { mapRef.current = map; }}
        />
        {overlays.length > 0 && (
          <div className="overlay-controls">
            <button className="primary" onClick={handleDownload}>Download Last Patch</button>
          </div>
        )}
      </div>
    </div>
  );
}
