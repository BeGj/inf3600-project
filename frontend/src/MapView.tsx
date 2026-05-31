import { useEffect, useRef } from "react";
import Map from "ol/Map";
import View from "ol/View";
import TileLayer from "ol/layer/Tile";
import ImageLayer from "ol/layer/Image";
import WebGLTileLayer from "ol/layer/WebGLTile";
import GeoTIFF from "ol/source/GeoTIFF";
import Static from "ol/source/ImageStatic";
import OSM from "ol/source/OSM";
import { transformExtent, toLonLat, fromLonLat } from "ol/proj";
import { containsExtent, intersects } from "ol/extent";
import { Draw, Link } from "ol/interaction";
import VectorLayer from "ol/layer/Vector";
import VectorSource from "ol/source/Vector";
import Feature from "ol/Feature";
import { fromExtent as polygonFromExtent } from "ol/geom/Polygon";
import GeoJSON from "ol/format/GeoJSON";
import type { Polygon } from "geojson";
import "ol/ol.css";

export interface ResultOverlay {
  maskId: string;
  image_b64: string;
  /** [lng_min, lat_min, lng_max, lat_max] WGS-84 */
  bbox: [number, number, number, number];
  prompt: string;
  negativePrompt: string;
  modelId: string;
  mask: Polygon;
  guidanceScale: number;
  strength: number;
  numInferenceSteps: number;
  visible: boolean;
}

interface MapViewProps {
  cogUrl: string | null;
  overlays: ResultOverlay[];
  drawingActive: boolean;
  clearKey: number;
  /** Footprint to preview on the map (hovered search result), WGS-84, or null. */
  highlightBBox: [number, number, number, number] | null;
  /** Polygon to show in the draw layer without entering draw mode (e.g. restored from Edit). */
  preloadMask?: Polygon | null;
  onMaskDrawn: (geojson: Polygon) => void;
  /** Called when the user tries to draw outside the loaded image. */
  onInvalidDraw?: (message: string) => void;
  /** Called once with the OpenLayers Map instance so the parent can read the view extent. */
  onMapReady?: (map: Map) => void;
}

const DRAW_STYLE = {
  "stroke-color": "#ff4444",
  "stroke-width": 2,
  "fill-color": "rgba(255,68,68,0.15)",
};

const HIGHLIGHT_STYLE = {
  "stroke-color": "#4a9eff",
  "stroke-width": 2,
  "stroke-line-dash": [6, 4],
  "fill-color": "rgba(74,158,255,0.08)",
};

export default function MapView({ cogUrl, overlays, drawingActive, clearKey, highlightBBox, preloadMask, onMaskDrawn, onInvalidDraw, onMapReady }: MapViewProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<Map | null>(null);
  const drawSource = useRef(new VectorSource());
  const highlightSource = useRef(new VectorSource());
  const drawInteraction = useRef<Draw | null>(null);
  const overlayLayers = useRef<Record<string, ImageLayer<Static>>>({});
  const overlayImages = useRef<Record<string, string>>({});
  // Extent of the loaded COG, in the map view projection. Drawing is restricted to it.
  const cogExtent = useRef<number[] | null>(null);
  // True once a COG has replaced the view with its own projection/resolutions.
  const viewIsCog = useRef(false);

  // Initialise map once
  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;

    const map = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({ source: new OSM(), zIndex: 0 }),
        new VectorLayer({ source: highlightSource.current, style: HIGHLIGHT_STYLE, zIndex: 5 }),
        new VectorLayer({ source: drawSource.current, style: DRAW_STYLE, zIndex: 10 }),
      ],
      view: new View({ center: [0, 0], zoom: 2, projection: "EPSG:3857" }),
    });
    mapInstance.current = map;

    // Sync the view (center x/y, zoom z, rotation r, layers l) to the URL so map state
    // is shareable/bookmarkable, and restore it on load. Re-binds itself when the view is
    // replaced (e.g. when a COG with its own projection loads).
    map.addInteraction(new Link());

    onMapReady?.(map);

    return () => {
      map.setTarget(undefined);
      mapInstance.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load COG layer when URL changes
  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !cogUrl) return;

    const cogLayer = new WebGLTileLayer({
      source: new GeoTIFF({
        sources: [{ url: cogUrl }],
        normalize: true,
        convertToRGB: 'auto',
      }),
      zIndex: 1,
    });
    map.addLayer(cogLayer);

    // Don't adopt the COG's `extent` — keeping the extent would lock panning to the image.
    // We capture the extent separately to restrict drawing.
    (cogLayer.getSource() as GeoTIFF).getView().then((viewOptions) => {
      const opts = viewOptions as typeof viewOptions & { extent?: number[]; resolutions?: number[]; projection?: unknown };
      const { extent, resolutions, ...viewRest } = opts;
      cogExtent.current = extent ?? null;

      // The COG's native resolutions cap zoom-in at the finest overview. Append a few
      // finer levels so users can zoom past native resolution (imagery oversamples).
      let res = resolutions;
      if (res && res.length > 0) {
        const finest = res[res.length - 1];
        res = [...res, finest / 2, finest / 4, finest / 8];
      }

      // Decide whether to keep the user's current view: only move the map if the selected
      // image is outside the current view. Compare both extents in WGS-84.
      const oldView = map.getView();
      const curExtent4326 = transformExtent(oldView.calculateExtent(map.getSize()), oldView.getProjection(), "EPSG:4326");
      const newProj = (viewRest as { projection?: Parameters<typeof transformExtent>[2] }).projection;
      const cogExtent4326 = extent && newProj ? transformExtent(extent, newProj, "EPSG:4326") : null;
      const overlaps = cogExtent4326 ? intersects(curExtent4326, cogExtent4326) : false;

      const newView = new View({ ...viewRest, resolutions: res });
      map.setView(newView);
      viewIsCog.current = true;

      if (overlaps && newProj) {
        // Image is in view — reproduce the user's current area in the new projection so the
        // map doesn't jump or zoom out.
        newView.fit(transformExtent(curExtent4326, "EPSG:4326", newProj), { duration: 0 });
      } else if (extent) {
        // Image is elsewhere — guide the user to it.
        newView.fit(extent, { duration: 600, padding: [40, 40, 40, 40] });
      }
    });

    return () => {
      map.removeLayer(cogLayer);
      cogExtent.current = null;
    };
  }, [cogUrl]);

  // When the scene is cleared, drop the COG's projection/zoom limits by restoring a
  // default, unconstrained Web Mercator view (kept at the current location).
  useEffect(() => {
    const map = mapInstance.current;
    if (!map || cogUrl || !viewIsCog.current) return;
    const old = map.getView();
    const center = old.getCenter();
    const lonLat = center ? toLonLat(center, old.getProjection()) : [0, 0];
    map.setView(
      new View({
        center: fromLonLat(lonLat, "EPSG:3857"),
        zoom: old.getZoom() ?? 2,
        projection: "EPSG:3857",
      })
    );
    viewIsCog.current = false;
    cogExtent.current = null;
  }, [cogUrl]);

  // Sync result overlays — reconcile rather than recreate so toggling visibility
  // on one overlay does not flicker the others.
  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;

    const viewProj = map.getView().getProjection();
    const incomingIds = new Set(overlays.map((o) => o.maskId));

    // Remove layers whose overlay has been deleted.
    for (const maskId of Object.keys(overlayLayers.current)) {
      if (!incomingIds.has(maskId)) {
        map.removeLayer(overlayLayers.current[maskId]);
        delete overlayLayers.current[maskId];
        delete overlayImages.current[maskId];
      }
    }

    // Add new layers; update visibility on existing ones (no recreate = no flicker).
    // If the same maskId has a new image (re-generate), replace the layer.
    overlays.forEach(({ maskId, image_b64, bbox, visible }) => {
      const existingLayer = overlayLayers.current[maskId];
      if (existingLayer && overlayImages.current[maskId] === image_b64) {
        existingLayer.setVisible(visible);
      } else {
        if (existingLayer) map.removeLayer(existingLayer);
        const [lngMin, latMin, lngMax, latMax] = bbox;
        const extent = transformExtent([lngMin, latMin, lngMax, latMax], "EPSG:4326", viewProj);
        const layer = new ImageLayer({
          source: new Static({
            url: `data:image/png;base64,${image_b64}`,
            imageExtent: extent,
          }),
          opacity: 1,
          zIndex: 2,
          visible,
        });
        map.addLayer(layer);
        overlayLayers.current[maskId] = layer;
        overlayImages.current[maskId] = image_b64;
      }
    });
  }, [overlays]);

  // Preview the hovered search result's footprint (no map movement on hover).
  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;
    highlightSource.current.clear();
    if (!highlightBBox) return;
    const extent = transformExtent(highlightBBox, "EPSG:4326", map.getView().getProjection());
    highlightSource.current.addFeature(new Feature(polygonFromExtent(extent)));
  }, [highlightBBox]);

  // Show a pre-existing mask polygon in the draw layer (e.g. restored via Edit).
  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;
    drawSource.current.clear();
    if (!preloadMask) return;
    const format = new GeoJSON();
    const viewProj = map.getView().getProjection();
    const result = format.readFeature(
      { type: "Feature", geometry: preloadMask, properties: {} },
      { featureProjection: viewProj, dataProjection: "EPSG:4326" }
    );
    const feature = Array.isArray(result) ? result[0] : result;
    drawSource.current.addFeature(feature);
  }, [preloadMask]);

  // Clear drawn polygon when clearKey increments
  useEffect(() => {
    if (clearKey > 0) drawSource.current.clear();
  }, [clearKey]);

  // Toggle polygon drawing interaction
  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;

    if (drawingActive) {
      drawSource.current.clear();
      const interaction = new Draw({ source: drawSource.current, type: "Polygon" });
      interaction.on("drawend", (evt) => {
        map.removeInteraction(interaction);

        // Reject polygons that extend beyond the loaded image — painting outside it
        // has no source imagery to inpaint against.
        const geomExtent = evt.feature.getGeometry()?.getExtent();
        if (cogExtent.current && geomExtent && !containsExtent(cogExtent.current, geomExtent)) {
          drawSource.current.clear();
          onInvalidDraw?.("Draw inside the loaded image.");
          return;
        }

        const format = new GeoJSON();
        const viewProj = map.getView().getProjection();
        const geojson = JSON.parse(
          format.writeFeature(evt.feature, { featureProjection: viewProj, dataProjection: "EPSG:4326" })
        ) as { geometry: Polygon };
        onMaskDrawn(geojson.geometry);
      });
      map.addInteraction(interaction);
      drawInteraction.current = interaction;
    } else {
      if (drawInteraction.current) {
        map.removeInteraction(drawInteraction.current);
        drawInteraction.current = null;
      }
    }
  }, [drawingActive, onMaskDrawn, onInvalidDraw]);

  return <div ref={mapRef} style={{ width: "100%", height: "100%" }} />;
}
