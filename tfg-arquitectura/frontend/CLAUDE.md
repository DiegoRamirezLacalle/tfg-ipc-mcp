# Frontend — Claude instructions

This is the React/Vite frontend for the TFG inflation-forecasting platform.

## Design source of truth

When `design.md` exists at the root of `tfg-arquitectura/frontend/`, **read it first** before
generating or modifying any UI component. It is the design system exported from Google Stitch
and contains:

- Colour tokens (semantic → hex)
- Typography scale and font families
- Spacing scale
- Component patterns + states
- Border radius and shadow tokens

**Rules:**
- Use only colours, fonts, spacings, and radii defined in `design.md`.
- Do not invent new values, do not fall back to Tailwind defaults.
- If a token from `design.md` is missing in `tailwind.config.js` or `src/index.css`,
  add it there first, then use it.
- When Stitch updates `design.md`, the diff in `src/index.css` (CSS variables under
  `:root` / `.dark`) and `tailwind.config.js` (extended theme) is the canonical refresh path.

## Component installation

For shadcn/ui components, prefer the project-local **shadcn MCP server** (configured in
`.mcp.json`) over manual file creation. After running `/mcp`, you can ask it to install
components by name and they will be placed in `src/components/ui/` automatically.

## Stack

- Vite + React 18 + TypeScript
- Tailwind v3 (with v4-compatible theme variables; upgrade path noted)
- shadcn/ui (style: new-york, baseColor: zinc)
- TanStack Query for all server state
- React Router v6
- Recharts for the forecast plots
- `@/` import alias → `src/`

## API client

- All hooks live in `src/lib/queries.ts`
- Types mirror `app/schemas/*.py` and live in `src/lib/types.ts`
- Auth token is stored in `localStorage` under `tfg_token`
- The dev server proxies `/api/*` to the FastAPI backend (`vite.config.ts`)

## Routes

| Path | Component | Backend endpoint(s) |
|---|---|---|
| `/login` | `Login` | `POST /auth/login` |
| `/experiments` | `ExperimentsList` | `GET /experiments` |
| `/experiments/new` | `NewExperiment` | `GET /datasets`, `GET /models`, `POST /experiments` |
| `/experiments/:id` | `ExperimentDetail` | `GET /experiments/:id`, `GET /experiments/:id/runs`, `POST /experiments/:id/runs` |
| `/runs/:id` | `RunDetail` | `GET /runs/:id`, `GET /runs/:id/predictions`, `GET /runs/:id/metrics`, `GET /runs/:id/mcp-context` |
| `/compare` | `Comparison` | `GET /metrics/compare?experiment_ids=...` |

## When the user pastes a Stitch export

1. Save `design.md` to `tfg-arquitectura/frontend/design.md`.
2. Translate its colour / spacing / radius tokens into:
   - `src/index.css` (CSS variables under `:root` and `.dark`)
   - `tailwind.config.js` (any custom font families or spacings)
3. Stitch's React export goes to `src/components/stitch/` — adapt screen-by-screen
   into the existing routed pages, keeping the TanStack Query hooks intact.
4. Replace placeholder layouts in `src/pages/*.tsx` with the Stitch-derived structure.
5. Verify against the FastAPI types in `src/lib/types.ts` — never invent fields.
