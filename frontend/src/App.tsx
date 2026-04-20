import { useState, useCallback } from "react";
import type { Polygon } from "geojson";
import MapView from "./MapView";
import type { ResultOverlay } from "./MapView";
import PromptPanel from "./PromptPanel";
import type { Mode } from "./PromptPanel";
import { inpaint, cloudRemove } from "./api";
import type { BBox } from "./api";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState<Mode>("sdxl");
  const [cogUrl, setCogUrl] = useState("");
  const [drawingActive, setDrawingActive] = useState(false);
  const [mask, setMask] = useState<Polygon | null>(null);
  const [overlays, setOverlays] = useState<ResultOverlay[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clearKey, setClearKey] = useState(0);

  // Rough WGS-84 bbox from the current map view — in a real app this would
  // come from the OpenLayers map's getView().calculateExtent() transformed to 4326.
  // For now the backend crops whatever bbox the mask polygon spans.
  const bboxFromMask = useCallback((geojson: Polygon): BBox => {
    const positions = geojson.coordinates.flat();
    if (positions.length === 0) return { lngMin: -180, latMin: -90, lngMax: 180, latMax: 90 };
    const lngs = positions.map((p) => p[0]);
    const lats = positions.map((p) => p[1]);
    return { lngMin: Math.min(...lngs), latMin: Math.min(...lats), lngMax: Math.max(...lngs), latMax: Math.max(...lats) };
  }, []);

  const handleMaskDrawn = useCallback((geojson: Polygon) => {
    setMask(geojson);
    setDrawingActive(false);
  }, []);

  const handleGenerate = useCallback(async (prompt: string) => {
    if (!cogUrl) return;
    setError(null);
    setLoading(true);
    try {
      let result;
      if (mode === "sdxl") {
        if (!mask) { setError("Draw a mask first."); setLoading(false); return; }
        const bbox = bboxFromMask(mask);
        result = await inpaint(bbox, mask, prompt, cogUrl);
      } else {
        // Cloud remove: use full image extent (backend will read the whole COG)
        const bbox: BBox = { lngMin: -180, latMin: -90, lngMax: 180, latMax: 90 };
        result = await cloudRemove(bbox, cogUrl);
      }
      setOverlays((prev) => [...prev, { image_b64: result.image_b64, bbox: result.bbox }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [cogUrl, mask, mode, bboxFromMask]);

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
      <PromptPanel
        mode={mode}
        onModeChange={setMode}
        cogUrl={cogUrl}
        onCogUrlChange={setCogUrl}
        drawingActive={drawingActive}
        maskReady={!!mask}
        onStartDraw={() => { setMask(null); setDrawingActive(true); setClearKey((k) => k + 1); }}
        onClearMask={() => { setMask(null); setDrawingActive(false); setClearKey((k) => k + 1); }}
        onGenerate={handleGenerate}
        loading={loading}
        error={error}
      />
      <div className="map-container">
        <MapView
          cogUrl={cogUrl || null}
          overlays={overlays}
          drawingActive={drawingActive}
          clearKey={clearKey}
          onMaskDrawn={handleMaskDrawn}
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
