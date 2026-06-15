# TFG Frontend

React + Vite + Tailwind interface for the inflation-forecasting platform.

## Local dev

```bash
npm install
npm run dev
# -> http://localhost:3000  (proxies /api -> http://localhost:8000)
```

## Docker

```bash
docker compose up frontend
# -> http://localhost:3000
```

## Structure

- `src/styles/` - design tokens (three themes), base styles, component classes
- `src/pages/` - routed pages; `src/components/` - charts, comparison views, visuals
- `src/lib/queries.ts` - all TanStack Query hooks; `src/lib/types.ts` mirrors the
  backend Pydantic schemas

Design mockups and the exported design system live in `docs/design/stitch/`
at the repository root.
