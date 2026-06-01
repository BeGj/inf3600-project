import { useEffect, useState } from "react";
import type { InpaintOptions, ModelInfo } from "./api";

// Backend defaults (backend/app/inference/pipeline.py) — kept in sync so the sliders
// start where the model would run if left untouched.
const DEFAULT_GUIDANCE_SCALE = 6.5;
const DEFAULT_STRENGTH = 1.0;
const DEFAULT_NUM_INFERENCE_STEPS = 40;

// FLUX.1-Fill-dev is guidance-distilled: no negative prompt / strength, and embedded
// guidance runs much higher than SD1.5. See backend/app/inference/pipeline.py (FLUX_*).
const FLUX_GUIDANCE_SCALE = 30;
const FLUX_NUM_INFERENCE_STEPS = 50;

export interface RestoreSnapshot {
  key: number;
  prompt: string;
  negativePrompt: string;
  guidanceScale: number;
  strength: number;
  numInferenceSteps: number;
}

interface PromptPanelProps {
  models: ModelInfo[];
  modelId: string;
  onModelChange: (id: string) => void;
  sceneReady: boolean;
  drawingActive: boolean;
  maskReady: boolean;
  onStartDraw: () => void;
  onClearMask: () => void;
  onGenerate: (prompt: string, opts: InpaintOptions) => void;
  loading: boolean;
  statusMessage: string | null;
  error: string | null;
  restoreSnapshot?: RestoreSnapshot | null;
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
  statusMessage,
  error,
  restoreSnapshot,
}: PromptPanelProps) {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [guidanceScale, setGuidanceScale] = useState(DEFAULT_GUIDANCE_SCALE);
  const [strength, setStrength] = useState(DEFAULT_STRENGTH);
  const [numInferenceSteps, setNumInferenceSteps] = useState(
    DEFAULT_NUM_INFERENCE_STEPS,
  );
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    if (!restoreSnapshot) return;
    setPrompt(restoreSnapshot.prompt);
    setNegativePrompt(restoreSnapshot.negativePrompt);
    setGuidanceScale(restoreSnapshot.guidanceScale);
    setStrength(restoreSnapshot.strength);
    setNumInferenceSteps(restoreSnapshot.numInferenceSteps);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [restoreSnapshot?.key]);

  // When the selected model changes, seed the prompt + negative prompt with its
  // defaults (unless the user has already typed something), and reset the inference
  // sliders to that family's defaults so the values match what the backend will use.
  const selected = models.find((m) => m.id === modelId);
  const isFlux = selected?.family === "flux-fill";
  useEffect(() => {
    if (selected && prompt.trim() === "") setPrompt(selected.default_prompt);
    if (selected && negativePrompt.trim() === "")
      setNegativePrompt(selected.negative_prompt);
    if (selected) {
      setGuidanceScale(
        selected.family === "flux-fill"
          ? FLUX_GUIDANCE_SCALE
          : DEFAULT_GUIDANCE_SCALE,
      );
      setNumInferenceSteps(
        selected.family === "flux-fill"
          ? FLUX_NUM_INFERENCE_STEPS
          : DEFAULT_NUM_INFERENCE_STEPS,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId]);

  return (
    <div
      style={{
        padding: "1rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.75rem",
        color: "#eee",
      }}
    >
      <h2 style={{ margin: 0, fontSize: "1rem" }}>Generative Fill</h2>

      {/* Model selector — driven by GET /models */}
      <label style={{ fontSize: "0.8rem" }}>
        Model
        <select
          value={modelId}
          onChange={(e) => onModelChange(e.target.value)}
          style={inputStyle}
        >
          {models.length === 0 && <option value="">Loading…</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id} disabled={!m.available}>
              {!m.available && m.disabled_reason ? m.disabled_reason : m.label}
            </option>
          ))}
        </select>
      </label>

      {/* Mask controls */}
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button
          onClick={onStartDraw}
          disabled={drawingActive || loading || !sceneReady}
          style={{
            flex: 1,
            padding: "0.4rem",
            background: drawingActive ? "#555" : "#2a6",
            border: "none",
            borderRadius: 4,
            color: "#fff",
            cursor: "pointer",
          }}
        >
          {drawingActive ? "Drawing…" : "Draw Mask"}
        </button>
        <button
          onClick={onClearMask}
          disabled={loading || (!maskReady && !drawingActive)}
          style={{
            padding: "0.4rem 0.8rem",
            background: "#555",
            border: "none",
            borderRadius: 4,
            color: "#fff",
            cursor: "pointer",
          }}
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

      {/* FLUX is guidance-distilled — it ignores negative prompts, so hide the field. */}
      {!isFlux && (
        <label style={{ fontSize: "0.8rem" }}>
          Negative prompt
          <textarea
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            rows={2}
            placeholder="Things to avoid in the result"
            style={{ ...inputStyle, resize: "vertical" }}
          />
        </label>
      )}

      {/* Advanced inference options */}
      <button
        onClick={() => setShowAdvanced((v) => !v)}
        style={{
          alignSelf: "flex-start",
          padding: 0,
          background: "none",
          border: "none",
          color: "#4a9eff",
          cursor: "pointer",
          fontSize: "0.75rem",
        }}
      >
        {showAdvanced ? "▾ Advanced options" : "▸ Advanced options"}
      </button>
      {showAdvanced && (
        <div
          style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}
        >
          <label style={{ fontSize: "0.75rem" }}>
            Guidance scale (prompt adherence): {guidanceScale}
            <input
              type="range"
              min={1}
              max={isFlux ? 50 : 20}
              step={0.5}
              value={guidanceScale}
              onChange={(e) => setGuidanceScale(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>
          {/* FluxFillPipeline doesn't use `strength`; hide it for FLUX. */}
          {!isFlux && (
            <label style={{ fontSize: "0.75rem" }}>
              Strength (how much to alter the region): {strength}
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={strength}
                onChange={(e) => setStrength(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </label>
          )}
          <label style={{ fontSize: "0.75rem" }}>
            Inference steps: {numInferenceSteps}
            <input
              type="range"
              min={10}
              max={80}
              step={1}
              value={numInferenceSteps}
              onChange={(e) => setNumInferenceSteps(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>
        </div>
      )}

      <button
        onClick={() =>
          onGenerate(prompt, {
            negativePrompt,
            guidanceScale,
            strength,
            numInferenceSteps,
          })
        }
        disabled={
          loading || !sceneReady || !maskReady || !prompt.trim() || !modelId
        }
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

      {!sceneReady && (
        <p style={{ margin: 0, fontSize: "0.75rem", color: "#aaa" }}>
          Search and select a scene first.
        </p>
      )}
      {sceneReady && maskReady && !loading && (
        <p style={{ margin: 0, fontSize: "0.75rem", color: "#8f8" }}>
          Mask ready.
        </p>
      )}
      {loading && statusMessage && (
        <p
          style={{
            margin: 0,
            fontSize: "0.75rem",
            color: statusMessage.startsWith("Downloading") ? "#f0a500" : "#ccc",
            lineHeight: 1.4,
          }}
        >
          {statusMessage}
        </p>
      )}
      {error && (
        <p style={{ margin: 0, fontSize: "0.75rem", color: "#f88" }}>{error}</p>
      )}
    </div>
  );
}
