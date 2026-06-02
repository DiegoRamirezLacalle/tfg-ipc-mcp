# TFG Frontend

React + Vite + Tailwind + shadcn/ui interface for the inflation-forecasting platform.

## Local dev

```bash
npm install
npm run dev
# → http://localhost:3000  (proxies /api → http://localhost:8000)
```

## Docker

```bash
docker compose up frontend
# → http://localhost:3000
```

## Stitch integration

1. Generate the design at https://stitch.withgoogle.com — describe the platform
   (experiments list, run detail, comparison table, dark mode).
2. Export `design.md` and the React components.
3. Drop `design.md` in this folder.
4. Ask Claude Code to read `CLAUDE.md` and translate tokens + screens.

## shadcn components

The shadcn MCP server is configured in `.mcp.json`. From a Claude Code session
inside this folder:

```
/mcp
# then: "install the shadcn button, card, table, badge, and dialog components"
```

Components land in `src/components/ui/` and use the tokens from `src/index.css`.
