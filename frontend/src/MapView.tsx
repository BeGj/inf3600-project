import { useEffect, useRef } from "react";
import Map from "ol/Map";
import View from "ol/View";
import TileLayer from "ol/layer/Tile";
import ImageLayer from "ol/layer/Image";
import WebGLTileLayer from "ol/layer/WebGLTile";
import GeoTIFF from "ol/source/GeoTIFF";
import Static from "ol/source/ImageStatic";
import OSM from "ol/source/OSM";
import { transformExtent } from "ol/proj";
import { Draw } from "ol/interaction";
import VectorLayer from "ol/layer/Vector";
import VectorSource from "ol/source/Vector";
import GeoJSON from "ol/format/GeoJSON";
import type { Polygon } from "geojson";
import "ol/ol.css";

export interface ResultOverlay {
  image_b64: string;
  /** [lng_min, lat_min, lng_max, lat_max] WGS-84 */
  bbox: [number, number, number, number];
}

interface MapViewProps {
  cogUrl: string | null;
  overlays: ResultOverlay[];
  drawingActive: boolean;
  clearKey: number;
  onMaskDrawn: (geojson: Polygon) => void;
}

const DRAW_STYLE = {
  "stroke-color": "#ff4444",
  "stroke-width": 2,
  "fill-color": "rgba(255,68,68,0.15)",
};

export default function MapView({ cogUrl, overlays, drawingActive, clearKey, onMaskDrawn }: MapViewProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<Map | null>(null);
  const drawSource = useRef(new VectorSource());
  const drawInteraction = useRef<Draw | null>(null);
  const overlayLayers = useRef<ImageLayer<Static>[]>([]);

  // Initialise map once
  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;

    const map = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({ source: new OSM(), zIndex: 0 }),
        new VectorLayer({ source: drawSource.current, style: DRAW_STYLE, zIndex: 10 }),
      ],
      view: new View({ center: [0, 0], zoom: 2, projection: "EPSG:3857" }),
    });
    mapInstance.current = map;

    return () => {
      map.setTarget(undefined);
      mapInstance.current = null;
    };
  }, []);

  // Load COG layer when URL changes
  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !cogUrl) return;

    const cogLayer = new WebGLTileLayer({
      source: new GeoTIFF({
        sources: [{ url: cogUrl }],
        normalize: true,
      }),
      zIndex: 1,
    });
    map.addLayer(cogLayer);

    // Zoom to COG extent
    (cogLayer.getSource() as GeoTIFF).getView().then((viewOptions) => {
      map.setView(new View(viewOptions));
    });

    return () => {
      map.removeLayer(cogLayer);
    };
  }, [cogUrl]);

  // Sync result overlays
  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;

    // Remove old overlay layers
    overlayLayers.current.forEach((l) => map.removeLayer(l));
    overlayLayers.current = [];

    overlays.forEach(({ image_b64, bbox }) => {
      const [lngMin, latMin, lngMax, latMax] = bbox;
      const extent = transformExtent([lngMin, latMin, lngMax, latMax], "EPSG:4326", "EPSG:3857");
      const layer = new ImageLayer({
        source: new Static({
          url: `data:image/png;base64,${image_b64}`,
          imageExtent: extent,
        }),
        opacity: 0.95,
        zIndex: 2,
      });
      map.addLayer(layer);
      overlayLayers.current.push(layer);
    });
  }, [overlays]);

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
        const format = new GeoJSON();
        const geojson = JSON.parse(format.writeFeature(evt.feature, { featureProjection: "EPSG:3857", dataProjection: "EPSG:4326" })) as { geometry: Polygon };
        onMaskDrawn(geojson.geometry);
        map.removeInteraction(interaction);
      });
      map.addInteraction(interaction);
      drawInteraction.current = interaction;
    } else {
      if (drawInteraction.current) {
        map.removeInteraction(drawInteraction.current);
        drawInteraction.current = null;
      }
    }
  }, [drawingActive, onMaskDrawn]);

  return <div ref={mapRef} style={{ width: "100%", height: "100%" }} />;
}
