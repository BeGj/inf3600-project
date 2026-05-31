import type { ResultOverlay } from "./MapView";

interface Props {
  overlays: ResultOverlay[];
  onRemove: (maskId: string) => void;
  onToggle: (maskId: string) => void;
  onZoom: (bbox: [number, number, number, number]) => void;
  onEdit: (overlay: ResultOverlay) => void;
}

export default function GeneratedImagesList({ overlays, onRemove, onToggle, onZoom, onEdit }: Props) {
  if (overlays.length === 0) return null;

  return (
    <div className="generated-list">
      <h3>Generated Images</h3>
      {overlays.map((overlay) => (
        <div key={overlay.maskId} className={`generated-list-item${overlay.visible ? "" : " hidden"}`}>
          <img
            src={`data:image/png;base64,${overlay.image_b64}`}
            alt="generated patch"
            style={{ opacity: overlay.visible ? 1 : 0.35 }}
          />
          <span className="label" title={overlay.prompt}>
            {overlay.prompt.length > 28 ? overlay.prompt.slice(0, 26) + "…" : overlay.prompt}
          </span>
          <div className="actions">
            <button
              onClick={() => onToggle(overlay.maskId)}
              title={overlay.visible ? "Hide on map" : "Show on map"}
              className={overlay.visible ? "active" : ""}
            >◉</button>
            <button onClick={() => onZoom(overlay.bbox)} title="Zoom to extent">⊕</button>
            <button onClick={() => onEdit(overlay)} title="Edit — restore mask and parameters">✏</button>
            <button onClick={() => onRemove(overlay.maskId)} title="Remove">×</button>
          </div>
        </div>
      ))}
    </div>
  );
}
