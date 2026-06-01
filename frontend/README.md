# Frontend — Satellite Generative Fill

React 19 + Vite + TypeScript + [OpenLayers](https://openlayers.org/) map UI. Lets users
search satellite imagery, stream a COG onto the map, draw a polygon, and request an
inpainted patch from the backend. See the [root README](../README.md) for the full
workflow and architecture.

## Run

```bash
npm ci
npm run dev      # http://localhost:5173
npm run build    # tsc type-check + production build (use this as the CI/lint gate)
npm run lint     # lint code using ESLint
npm run format     # format code using pretrtier
```

The backend base URL is read from `VITE_API_URL` (default `http://localhost:8000`); set it
in a `frontend/.env` file if your backend runs elsewhere.

## Source map (`src/`)

| File               | Responsibility                                                                  |
| ------------------ | ------------------------------------------------------------------------------- |
| `App.tsx`          | Top-level state + wiring: models, selected scene, mask, overlays.               |
| `api.ts`           | Typed backend client: `getModels`, `searchCatalog`, `inpaint`. All fetch calls. |
| `CatalogPanel.tsx` | Imagery search (date / cloud filters) over the current map view; scene list.    |
| `PromptPanel.tsx`  | Model `<select>`, mask draw/clear controls, prompt, Generate.                   |
| `MapView.tsx`      | OpenLayers map: COG streaming, polygon drawing, result overlays, view extent.   |

## How the pieces talk

- `MapView` streams the selected scene's `visual_href` COG directly from S3 via OpenLayers'
  `GeoTIFF` source — imagery does **not** pass through the backend.
- `MapView` hands its `Map` instance up via `onMapReady`; `App` uses it to compute the
  current view extent (WGS-84) for catalogue search.
- Drawing a polygon emits a GeoJSON `Polygon` (EPSG:4326). `App` derives its bbox and posts
  it with the prompt + `model_id` to `/inpaint`; the returned PNG is overlaid via
  `ImageStatic` at the same bbox.

## Conventions

- All backend calls go through `api.ts` — don't scatter `fetch` elsewhere.
- Coordinates crossing the API boundary are **WGS-84 (EPSG:4326)**; the map renders in
  EPSG:3857, so transform at the boundary (see `transformExtent` usage in `App`/`MapView`).
- `npm run build` must pass (it runs `tsc -b`) before committing.
