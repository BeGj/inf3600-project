import { useCallback, useEffect, useRef, useState } from "react";
import type { Polygon } from "geojson";
import Map from "ol/Map";
import { transformExtent } from "ol/proj";
import MapView from "./MapView";
import type { ResultOverlay } from "./MapView";
import PromptPanel from "./PromptPanel";
import CatalogPanel from "./CatalogPanel";
import { getModels, inpaint } from "./api";
import type { BBox, ModelInfo, Scene } from "./api";
import "./App.css";

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [scene, setScene] = useState<Scene | null>(null);
  const [drawingActive, setDrawingActive] = useState(false);
  const [mask, setMask] = useState<Polygon | null>(null);
  const [overlays, setOverlays] = useState<ResultOverlay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clearKey, setClearKey] = useState(0);
  const mapRef = useRef<Map | null>(null);

  // Load available models once.
  useEffect(() => {
    getModels()
      .then((m) => {
        setModels(m);
        if (m.length > 0) setModelId(m[0].id);
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
    setDrawingActive(false);
  }, []);

  const handleInvalidDraw = useCallback((msg: string) => {
    setError(msg);
    setDrawingActive(false);
  }, []);

  const handleSelectScene = useCallback((s: Scene) => {
    setScene(s);
    setMask(null);
    setError(null);
    setClearKey((k) => k + 1);
  }, []);

  const handleClearScene = useCallback(() => {
    setScene(null);
    setMask(null);
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
    async (prompt: string) => {
      if (!scene || !mask || !modelId) return;
      setError(null);
      setLoading(true);
      try {
        const bbox = bboxFromMask(mask);
        const result = await inpaint(bbox, mask, prompt, scene.visual_href, modelId);
        setOverlays((prev) => [...prev, { image_b64: result.image_b64, bbox: result.bbox }]);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    },
    [scene, mask, modelId, bboxFromMask]
  );

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
        <CatalogPanel getViewBBox={getViewBBox} selectedSceneId={scene?.id ?? null} onSelectScene={handleSelectScene} onClearScene={handleClearScene} onFitResults={fitToResults} />
        <PromptPanel
          models={models}
          modelId={modelId}
          onModelChange={setModelId}
          sceneReady={!!scene}
          drawingActive={drawingActive}
          maskReady={!!mask}
          onStartDraw={() => { setMask(null); setDrawingActive(true); setClearKey((k) => k + 1); }}
          onClearMask={() => { setMask(null); setDrawingActive(false); setClearKey((k) => k + 1); }}
          onGenerate={handleGenerate}
          loading={loading}
          error={error}
        />
      </div>
      <div className="map-container">
        <MapView
          cogUrl={scene?.visual_href ?? null}
          overlays={overlays}
          drawingActive={drawingActive}
          clearKey={clearKey}
          onMaskDrawn={handleMaskDrawn}
          onInvalidDraw={handleInvalidDraw}
          onMapReady={(map) => { mapRef.current = map; }}
        />
        {overlays.length > 0 && (
          <div className="overlay-controls">
            <button onClick={() => setOverlays([])}>Clear Overlays</button>
            <button className="primary" onClick={handleDownload}>Download Last Patch</button>
          </div>
        )}
      </div>
    </div>
  );
}
