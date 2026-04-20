import { useState } from "react";

export type Mode = "sdxl" | "prithvi";

interface PromptPanelProps {
  mode: Mode;
  onModeChange: (m: Mode) => void;
  cogUrl: string;
  onCogUrlChange: (url: string) => void;
  drawingActive: boolean;
  maskReady: boolean;
  onStartDraw: () => void;
  onClearMask: () => void;
  onGenerate: (prompt: string) => void;
  loading: boolean;
  error: string | null;
}

export default function PromptPanel({
  mode,
  onModeChange,
  cogUrl,
  onCogUrlChange,
  drawingActive,
  maskReady,
  onStartDraw,
  onClearMask,
  onGenerate,
  loading,
  error,
}: PromptPanelProps) {
  const [prompt, setPrompt] = useState("clear blue ocean");

  return (
    <div style={{ padding: "1rem", display: "flex", flexDirection: "column", gap: "0.75rem", background: "#1a1a1a", color: "#eee", width: 300, boxSizing: "border-box" }}>
      <h2 style={{ margin: 0, fontSize: "1rem" }}>Satellite Generative Fill</h2>

      {/* Mode toggle */}
      <div style={{ display: "flex", gap: "0.5rem" }}>
        {(["sdxl", "prithvi"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => onModeChange(m)}
            style={{
              flex: 1,
              padding: "0.4rem",
              background: mode === m ? "#4a9eff" : "#333",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontWeight: mode === m ? "bold" : "normal",
            }}
          >
            {m === "sdxl" ? "Generative Fill (SDXL)" : "Cloud Remove (Prithvi)"}
          </button>
        ))}
      </div>

      {/* COG URL input */}
      <label style={{ fontSize: "0.8rem" }}>
        COG URL
        <input
          value={cogUrl}
          onChange={(e) => onCogUrlChange(e.target.value)}
          placeholder="https://example.com/image.tif"
          style={{ width: "100%", marginTop: 4, padding: "0.4rem", background: "#333", color: "#eee", border: "1px solid #555", borderRadius: 4, boxSizing: "border-box" }}
        />
      </label>

      {/* Mask controls */}
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button
          onClick={onStartDraw}
          disabled={drawingActive || loading}
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

      {/* Prompt (SDXL only) */}
      {mode === "sdxl" && (
        <label style={{ fontSize: "0.8rem" }}>
          Prompt
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            style={{ width: "100%", marginTop: 4, padding: "0.4rem", background: "#333", color: "#eee", border: "1px solid #555", borderRadius: 4, resize: "vertical", boxSizing: "border-box" }}
          />
        </label>
      )}

      {/* Generate button */}
      <button
        onClick={() => onGenerate(prompt)}
        disabled={loading || !cogUrl || (mode === "sdxl" && (!maskReady || !prompt.trim()))}
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

      {/* Status / error */}
      {maskReady && !loading && <p style={{ margin: 0, fontSize: "0.75rem", color: "#8f8" }}>Mask ready.</p>}
      {error && <p style={{ margin: 0, fontSize: "0.75rem", color: "#f88" }}>{error}</p>}
    </div>
  );
}
