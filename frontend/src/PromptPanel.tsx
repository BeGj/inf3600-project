import { useEffect, useState } from "react";
import type { ModelInfo } from "./api";

interface PromptPanelProps {
  models: ModelInfo[];
  modelId: string;
  onModelChange: (id: string) => void;
  sceneReady: boolean;
  drawingActive: boolean;
  maskReady: boolean;
  onStartDraw: () => void;
  onClearMask: () => void;
  onGenerate: (prompt: string) => void;
  loading: boolean;
  error: string | null;
}

const inputStyle: React.CSSProperties = {
  width: "100%",
  marginTop: 4,
  padding: "0.4rem",
  background: "#333",
  color: "#eee",
  border: "1px solid #555",
  borderRadius: 4,
  boxSizing: "border-box",
};

export default function PromptPanel({
  models,
  modelId,
  onModelChange,
  sceneReady,
  drawingActive,
  maskReady,
  onStartDraw,
  onClearMask,
  onGenerate,
  loading,
  error,
}: PromptPanelProps) {
  const [prompt, setPrompt] = useState("");

  // When the selected model changes, seed the prompt with its default (unless the
  // user has already typed something).
  const selected = models.find((m) => m.id === modelId);
  useEffect(() => {
    if (selected && prompt.trim() === "") setPrompt(selected.default_prompt);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId]);

  return (
    <div style={{ padding: "1rem", display: "flex", flexDirection: "column", gap: "0.75rem", color: "#eee" }}>
      <h2 style={{ margin: 0, fontSize: "1rem" }}>Generative Fill</h2>

      {/* Model selector — driven by GET /models */}
      <label style={{ fontSize: "0.8rem" }}>
        Model
        <select value={modelId} onChange={(e) => onModelChange(e.target.value)} style={inputStyle}>
          {models.length === 0 && <option value="">Loading…</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </label>

      {/* Mask controls */}
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button
          onClick={onStartDraw}
          disabled={drawingActive || loading || !sceneReady}
          style={{ flex: 1, padding: "0.4rem", background: drawingActive ? "#555" : "#2a6", border: "none", borderRadius: 4, color: "#fff", cursor: "pointer" }}
        >
          {drawingActive ? "Drawing…" : "Draw Mask"}
        </button>
        <button
          onClick={onClearMask}
          disabled={loading || (!maskReady && !drawingActive)}
          style={{ padding: "0.4rem 0.8rem", background: "#555", border: "none", borderRadius: 4, color: "#fff", cursor: "pointer" }}
        >
          Clear
        </button>
      </div>

      <label style={{ fontSize: "0.8rem" }}>
        Prompt
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          style={{ ...inputStyle, resize: "vertical" }}
        />
      </label>

      <button
        onClick={() => onGenerate(prompt)}
        disabled={loading || !sceneReady || !maskReady || !prompt.trim() || !modelId}
        style={{
          padding: "0.6rem",
          background: loading ? "#555" : "#4a9eff",
          border: "none",
          borderRadius: 4,
          color: "#fff",
          fontWeight: "bold",
          cursor: "pointer",
          fontSize: "0.95rem",
        }}
      >
        {loading ? "Generating…" : "Generate"}
      </button>

      {!sceneReady && <p style={{ margin: 0, fontSize: "0.75rem", color: "#aaa" }}>Search and select a scene first.</p>}
      {sceneReady && maskReady && !loading && <p style={{ margin: 0, fontSize: "0.75rem", color: "#8f8" }}>Mask ready.</p>}
      {error && <p style={{ margin: 0, fontSize: "0.75rem", color: "#f88" }}>{error}</p>}
    </div>
  );
}
