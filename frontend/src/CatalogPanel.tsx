import { memo, useEffect, useState } from "react";
import { getCatalogs, getEvents, searchCatalog } from "./api";
import type { BBox, CatalogEvent, CatalogInfo, Scene } from "./api";

interface CatalogPanelProps {
  /** Returns the current map view extent in WGS-84, or null if the map isn't ready. */
  getViewBBox: () => BBox | null;
  selectedSceneId: string | null;
  onSelectScene: (scene: Scene) => void;
  /** Preview a result's footprint on the map while hovering it (null on mouse-out). */
  onHoverScene: (scene: Scene | null) => void;
  onClearScene: () => void;
  /** Fit the map to these scenes (used for event catalogues whose coverage is elsewhere). */
  onFitResults: (scenes: Scene[]) => void;
}

const fmtDate = (d: Date) => d.toISOString().slice(0, 10);

// Default search window: the last month up to today.
function defaultDateRange(): { start: string; end: string } {
  const end = new Date();
  const start = new Date(end);
  start.setMonth(start.getMonth() - 1);
  return { start: fmtDate(start), end: fmtDate(end) };
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

function CatalogPanel({
  getViewBBox,
  selectedSceneId,
  onSelectScene,
  onHoverScene,
  onClearScene,
  onFitResults,
}: CatalogPanelProps) {
  const initialRange = defaultDateRange();
  const [catalogs, setCatalogs] = useState<CatalogInfo[]>([]);
  const [catalogId, setCatalogId] = useState("");
  const [events, setEvents] = useState<CatalogEvent[]>([]);
  const [eventId, setEventId] = useState("");
  const [start, setStart] = useState(initialRange.start);
  const [end, setEnd] = useState(initialRange.end);
  const [maxCloud, setMaxCloud] = useState(20);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const catalog = catalogs.find((c) => c.id === catalogId);

  // Load catalogues once.
  useEffect(() => {
    getCatalogs()
      .then((cs) => {
        setCatalogs(cs);
        if (cs.length > 0) setCatalogId(cs[0].id);
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }, []);

  // Load events when an event-based catalogue is selected.
  useEffect(() => {
    setEventId("");
    setEvents([]);
    if (!catalog?.requires_event) return;
    getEvents(catalog.id)
      .then((evs) => {
        setEvents(evs);
        if (evs.length > 0) setEventId(evs[0].id);
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }, [catalogId]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSearch = async () => {
    if (!catalog) return;
    if (catalog.requires_event && !eventId) {
      setError("Pick an event first.");
      return;
    }
    const bbox = getViewBBox();
    if (!bbox) {
      setError("Map not ready yet.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const result = await searchCatalog({
        bbox,
        catalog: catalog.id,
        event: catalog.requires_event ? eventId : undefined,
        datetime: catalog.supports_datetime ? `${start}/${end}` : undefined,
        maxCloudCover: catalog.supports_cloud ? maxCloud : undefined,
        limit: 50,
      });
      result.sort((a, b) => (b.datetime ?? "").localeCompare(a.datetime ?? ""));
      setScenes(result);
      if (result.length === 0) {
        setError(
          catalog.requires_event
            ? "No tiles for this event in view — try zooming out."
            : "No scenes found for this view/filters.",
        );
      } else if (catalog.requires_event) {
        // Event imagery is elsewhere than the user's current view — fly there.
        onFitResults(result);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        padding: "1rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.6rem",
        color: "#eee",
        borderBottom: "1px solid #333",
      }}
    >
      <h2 style={{ margin: 0, fontSize: "1rem" }}>Find imagery</h2>

      <label style={{ fontSize: "0.75rem" }}>
        Catalogue
        <select
          value={catalogId}
          onChange={(e) => setCatalogId(e.target.value)}
          style={inputStyle}
        >
          {catalogs.length === 0 && <option value="">Loading…</option>}
          {catalogs.map((c) => (
            <option key={c.id} value={c.id}>
              {c.label}
            </option>
          ))}
        </select>
      </label>
      {catalog && (
        <p style={{ margin: 0, fontSize: "0.7rem", color: "#aaa" }}>
          ~{catalog.resolution_m} m · {catalog.coverage}
        </p>
      )}

      {catalog?.requires_event && (
        <label style={{ fontSize: "0.75rem" }}>
          Event
          <select
            value={eventId}
            onChange={(e) => setEventId(e.target.value)}
            style={inputStyle}
          >
            {events.length === 0 && <option value="">Loading…</option>}
            {events.map((ev) => (
              <option key={ev.id} value={ev.id}>
                {ev.label}
              </option>
            ))}
          </select>
        </label>
      )}

      {catalog?.supports_datetime && (
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <label style={{ fontSize: "0.75rem", flex: 1 }}>
            From
            <input
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              style={inputStyle}
            />
          </label>
          <label style={{ fontSize: "0.75rem", flex: 1 }}>
            To
            <input
              type="date"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              style={inputStyle}
            />
          </label>
        </div>
      )}

      {catalog?.supports_cloud && (
        <label style={{ fontSize: "0.75rem" }}>
          Max cloud cover: {maxCloud}%
          <input
            type="range"
            min={0}
            max={100}
            value={maxCloud}
            onChange={(e) => setMaxCloud(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </label>
      )}

      <button
        onClick={handleSearch}
        disabled={loading || !catalog}
        style={{
          padding: "0.5rem",
          background: loading ? "#555" : "#4a9eff",
          border: "none",
          borderRadius: 4,
          color: "#fff",
          fontWeight: "bold",
          cursor: "pointer",
        }}
      >
        {loading
          ? "Searching…"
          : catalog?.requires_event
            ? "Search event"
            : "Search this view"}
      </button>

      {error && (
        <p style={{ margin: 0, fontSize: "0.75rem", color: "#f88" }}>{error}</p>
      )}

      {selectedSceneId && (
        <button
          onClick={onClearScene}
          style={{
            padding: "0.4rem",
            background: "#444",
            border: "none",
            borderRadius: 4,
            color: "#fff",
            cursor: "pointer",
            fontSize: "0.8rem",
          }}
        >
          Clear selected image
        </button>
      )}

      {scenes.length > 0 && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.4rem",
            maxHeight: 260,
            overflowY: "auto",
          }}
        >
          {scenes.map((scene) => {
            const active = scene.id === selectedSceneId;
            return (
              <button
                key={scene.id}
                onClick={() => onSelectScene(scene)}
                onMouseEnter={() => onHoverScene(scene)}
                onMouseLeave={() => onHoverScene(null)}
                style={{
                  display: "flex",
                  gap: "0.5rem",
                  alignItems: "center",
                  textAlign: "left",
                  padding: "0.3rem",
                  background: active ? "#2a4a6a" : "#222",
                  border: active ? "1px solid #4a9eff" : "1px solid #333",
                  borderRadius: 4,
                  color: "#eee",
                  cursor: "pointer",
                }}
              >
                {scene.thumbnail && (
                  <img
                    src={scene.thumbnail}
                    alt=""
                    width={48}
                    height={48}
                    style={{ objectFit: "cover", borderRadius: 3 }}
                  />
                )}
                <span style={{ fontSize: "0.7rem", lineHeight: 1.3 }}>
                  {scene.datetime?.slice(0, 10) ?? "unknown date"}
                  <br />
                  {scene.cloud_cover != null
                    ? `${scene.cloud_cover.toFixed(0)}% cloud`
                    : ""}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default memo(CatalogPanel);
