# Stitch design brief — TFG Inflation Forecasting Platform

Paste the **main prompt** below into Stitch in one go. Then iterate per-screen using
the prompts in section 4. The visual references in section 1 are search queries you
can run on Dribbble to give Stitch a mood reference (Stitch accepts image URLs).

---

## 1 · Aesthetic direction

**One-line pitch:** *Bloomberg Terminal meets academic journal* — high data density,
scientific neutrality, monospace numerics, restrained colour, no decoration that
isn't carrying information.

**Visual mood references** (search and pick 2–3 images to attach to the Stitch prompt):

- Dribbble → "Linear dashboard dark mode" (Einstein redesign)
- Dribbble → "Posthog dashboard"
- Dribbble → "Weights & Biases compare runs"
- Dribbble → "Bloomberg terminal academic" (the "scientific neutrality" projects)
- Vercel's own dashboard (vercel.com → after sign-in, the project view)

What to copy from each:
- **Linear** → typography tightness, status badge density, hover states
- **Posthog** → the data tables with sparkline trends
- **W&B "compare runs"** → the metrics-table-as-product pattern (this is *the* view for our thesis)
- **Bloomberg** → monospace numeric columns, dense rows, zero ornament
- **Vercel** → the navigation bar simplicity, deploy-status badges

---

## 2 · Design tokens (give these to Stitch verbatim)

### Colour system — zinc-tinted near-black

```
Background system (dark mode primary)
  --background          #09090B   (page background — zinc-tinted black, not pure)
  --card                #0F0F12   (elevated surfaces, table rows)
  --muted               #18181B   (hover, secondary chips)
  --border              #27272A   (1px dividers, input borders)
  --border-strong       #3F3F46   (active states, focused borders)

Text
  --foreground          #FAFAFA   (primary)
  --foreground-muted    #A1A1AA   (secondary labels, timestamps)
  --foreground-subtle   #71717A   (placeholders, disabled)

Status palette (charts + badges)
  --success             #10B981   (done, completed runs)
  --info                #06B6D4   (running, active polling)
  --warning             #F59E0B   (pending, warning)
  --destructive         #EF4444   (failed, errors)
  --mcp                 #8B5CF6   (MCP-enriched runs — accent the use_mcp=true rows)

Chart series palette (model comparison — distinct, colour-blind safe)
  naive-seasonal        #E4E4E7   (zinc-200, the baseline reference)
  sarima                #06B6D4   (cyan)
  ridge-exog            #F59E0B   (amber)
  timesfm               #A78BFA   (violet)
  chronos-2             #34D399   (emerald)
  timegpt               #FB7185   (rose)
```

### Typography

```
Sans family  →  Geist Sans  (fallback: Inter, ui-sans-serif)
Mono family  →  Geist Mono  (fallback: JetBrains Mono, ui-monospace)

Use MONO for:
  - metric values (0.1234)
  - run IDs (#42)
  - timestamps (2024-03-15)
  - model slugs (sarima, ridge-exog)
  - any code-like identifier

Use SANS for everything else, with tight tracking on headers.

Scale
  Display    32px   font-weight 600  tracking -0.02em
  H1         24px   font-weight 600  tracking -0.015em
  H2         16px   font-weight 600  tracking -0.01em
  Body       14px   font-weight 400
  Small      13px   font-weight 400
  Micro      11px   font-weight 500  uppercase  tracking 0.08em
```

### Spacing & geometry

```
Border radius
  sm   4px   (chips, badges)
  md   6px   (buttons, inputs)
  lg   8px   (cards, panels)

Shadows — barely visible, technical
  shadow-sm   0 1px 0 0 rgba(255,255,255,0.04) inset
              (a hairline highlight on top of dark surfaces, no blur)

Spacing scale  4 / 8 / 12 / 16 / 24 / 32 / 48 / 64

Grid       12-column, 24px gutter on desktop, container max-width 1400px
```

---

## 3 · Layout pattern

**Top navigation bar (sticky, 56px tall, border-bottom):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  tfg-ipc-mcp     Experiments  New  Compare              user@email · admin │
└─────────────────────────────────────────────────────────────────────────────┘
```

- Left: brand wordmark in **Geist Mono**, lowercase, no logo icon
- Center: pill nav items, active state = filled secondary background
- Right: user email + role chip (small uppercase) + sign-out

**Content area:** 64px top padding, container 1400px max, single column on most pages,
sidebar+main layout on Comparison.

---

## 4 · Six screens — exact specifications

### Screen 1 · Login

- Full-page centered card, 384px wide
- Card has 32px padding, `border-border`, `bg-card`
- Heading "TFG Forecasting Platform" (display size)
- Subheading "Sign in to continue" (small, muted)
- Email + password inputs (h-9, monospace input for email since users are technical)
- Primary button full-width
- No "forgot password", no "sign up with Google" — this is an academic demo
- A small footer line "Backend status · ●" with the colour from `/health` (cyan = ok, red = degraded)

### Screen 2 · Experiments list

- Page header: H1 "Experiments" + small muted subtitle "12 total · 8 completed"
- Right side: filter pills (All · Completed · Running · Failed) + "New experiment" primary button
- Data table fills the rest:
  - Columns: Name · Model · Horizon · MCP · Status · Last run · Created
  - Status column: filled badge with the status palette colour at 15% opacity, text at 100%
  - Model column: monospace slug
  - "MCP" column: just a violet dot (`--mcp`) if true, em-dash if false
  - Row hover: `bg-muted` with cursor-pointer
  - First-row treatment: no top border (the header has bottom border)
- Empty state: dashed border card, "No experiments yet — create your first one"

### Screen 3 · New experiment

- Two-column layout on desktop: form on left (max-w 560px), live preview on right
- Form sections separated by 24px gaps, no card backgrounds — flat against page
- Cascading selects: Dataset → Series (disabled until dataset picked) → Model
- Model select shows model name + slug in mono (e.g., "SARIMA · sarima")
- Horizon: number input + small inline range slider 1-60
- "Use MCP semantic context" — toggle switch, NOT a checkbox, with violet active colour
  - Below it: tiny muted text "Fetches ECB/FOMC signals from the MCP server for the
    forecast period and attaches them to the run for reproducibility."
- Right column "Summary": shows the JSON payload that will be POSTed (read-only,
  monospace, syntax-highlighted) — demystifies what the API receives. This is a
  research-tool flex, like Postman's request preview.
- Primary button at bottom: "Create experiment" (right-aligned)

### Screen 4 · Experiment detail

- Header: H1 = experiment name, monospace metadata strip below
  `series → indice_general · horizon=12 · use_mcp=true · created 2024-03-15 14:30`
- Action bar (right-aligned): "Trigger new run" primary, "Delete" ghost destructive
- Two stacked sections:
  1. **Run history** — same status-badge table pattern as experiments list
     - Columns: Run · Status · Started · Finished · MAE · RMSE · MAPE · ▸
     - Click row → /runs/:id
     - Pending/running rows have a subtle 2-second pulse animation on the status badge
  2. **Forecast comparison panel** (if ≥1 done run) — shows the most recent done run's
     predictions overlaid on the recent history of the actual series, line chart 320px tall

### Screen 5 · Run detail  *(this is the page that needs to look excellent — most-screenshotted)*

- Top breadcrumb: `experiments / SARIMA Spain IPC h=12 / run #42`
- Header: `Run #42` mono large + status badge inline + finished_at small muted on right
- **Three metric cards in a row** (the hero):

  ```
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ MAE              │  │ RMSE             │  │ MAPE             │
  │                  │  │                  │  │                  │
  │ 0.1234           │  │ 0.1856           │  │  2.34%           │
  │ mono · 32px      │  │ mono · 32px      │  │ mono · 32px      │
  │                  │  │                  │  │                  │
  │ ↓ 12% vs naive   │  │ ↓ 18% vs naive   │  │ ↓  9% vs naive   │
  │ tiny muted       │  │ tiny muted       │  │ tiny muted       │
  └──────────────────┘  └──────────────────┘  └──────────────────┘
  ```

  Card: `bg-card`, `border-border`, 24px padding, h-40
  The "vs naive" comparison gets a tiny coloured arrow (success/destructive)

- **Forecast chart** — full-width, 400px tall
  - Two lines: actuals (white, solid), predictions (model colour, solid)
  - Optional shaded band for confidence intervals if `lower_ci`/`upper_ci` present
  - Vertical dashed line at the forecast origin (start of test set)
  - X axis: monthly ticks "2024-01", "2024-02", … (mono)
  - Y axis: 4 ticks, mono, no grid lines (rely on horizontal hair-lines at axis only)
  - Tooltip: dark card with mono numbers, no animation, instant

- **MCP context** (only if mcp-context endpoint returns 200)
  - Collapsible card titled "MCP context · 12 signals" with violet `●` dot
  - When expanded: small two-column key-value list of signals + a "Raw JSON" toggle
    that flips the body to a `<pre>` view

- **Predictions table** (bottom, collapsed by default)
  - "Show 12 predictions ▾"
  - When expanded: simple two-column mono table — timestamp / value

### Screen 6 · Comparison  *(the thesis-defining view)*

This is the W&B "compare runs" pattern but tighter.

- **Left sidebar** (260px, sticky):
  - "Experiments" micro label uppercase
  - Search box (filter by name)
  - Checkbox list of experiments, grouped by model slug — collapsible groups
  - Each row: checkbox + name + tiny model-coloured dot
  - Footer: "Compare N selected" count + a "Clear" link

- **Main panel:**
  - H1 "Comparison" + count chip "3 experiments"
  - **Big chart at top** (480px tall): all selected experiments' predictions on the
    same axes, coloured by model from the chart palette
  - **Leaderboard table below**:
    - Columns: ● · Experiment · Model · Horizon · MCP · MAE · RMSE · MAPE · Run
    - Leading dot uses the chart series colour (so chart ↔ table mapping is obvious)
    - Best MAE / RMSE / MAPE cells are bolded and underlined dotted
    - Rows are sortable (click header), default sort: MAE ascending
  - Optional below the table: "Export as LaTeX" ghost button that copies a markdown
    table to clipboard. The PhD-thesis wink.

---

## 5 · Anti-patterns (paste these too — Stitch tends toward defaults)

> **Do not:**
> - use gradients, glassmorphism, or "glow" effects
> - use sans-serif for numbers
> - use rounded-full pills for status — they should be `rounded` (6px) rectangles
> - decorate the empty states with illustrations
> - add a "Welcome back" personal greeting anywhere
> - use Material Design or Ant Design icons — use Lucide only
> - pad cards more than 24px
> - use saturated colours for backgrounds; only for accents
> - show colour without information (no "decoration")

---

## 6 · Final prompt block (the literal text to paste into Stitch)

```
Design a 6-screen web app called the TFG Inflation Forecasting Platform — a research
tool for running and comparing machine-learning forecasts on Spanish inflation
(IPC) time series. The aesthetic is "Bloomberg Terminal meets academic journal":
dark, monospace-heavy, data-dense, no decoration that isn't information.

Visual references: Linear's dashboard dark mode, PostHog tables, Weights & Biases
"compare runs" page, Vercel's project view, Bloomberg Terminal grids.

Theme: dark mode only (no light variant needed). Use these exact tokens:
- Background: #09090B (zinc-tinted near-black)
- Card surface: #0F0F12
- Muted/hover: #18181B
- Border: #27272A
- Text primary: #FAFAFA, muted: #A1A1AA
- Success: #10B981, Info: #06B6D4, Warning: #F59E0B, Destructive: #EF4444
- MCP accent (special): #8B5CF6 violet
- Chart series: white #E4E4E7, cyan #06B6D4, amber #F59E0B, violet #A78BFA,
  emerald #34D399, rose #FB7185

Typography: Geist Sans for UI, Geist Mono for ALL numeric values, run IDs,
timestamps, model slugs, and any code-like identifier. Tight tracking on headers.

Radii: 4 / 6 / 8 px only. No large rounded pills, no rounded-full.
Spacing scale: 4, 8, 12, 16, 24, 32, 48, 64.

Six screens:
  1. Login — centered 384px card, email + password, no social sign-in, no signup
  2. Experiments list — top nav, data table with status badges, filter pills,
     a New experiment primary button. Status colours from the palette at 15%
     opacity background, 100% text.
  3. New experiment — two columns: form on left (cascading Dataset → Series →
     Model selects, horizon number input with slider, MCP toggle in violet),
     live JSON request-payload preview on right (read-only, monospace).
  4. Experiment detail — header with monospace metadata strip, runs table with
     inline metrics columns, a forecast comparison line chart at the bottom.
  5. Run detail — breadcrumb, three big metric cards (MAE / RMSE / MAPE with
     "vs naive" deltas), a full-width 400px forecast line chart with actuals
     (white solid) and predictions (model colour solid), a collapsible MCP
     context panel (only if MCP data exists), a collapsible predictions table.
  6. Comparison — 260px left sidebar with a grouped checkbox list of experiments,
     main panel with a big multi-series prediction chart and a sortable
     leaderboard table where the best metric per column is bolded.

Top navigation: sticky 56px bar, brand "tfg-ipc-mcp" in Geist Mono lowercase on
the left, three pill nav items in the centre (Experiments, New, Compare),
user email + uppercase role chip + sign-out on the right.

Do not use gradients, glassmorphism, glow effects, illustrations in empty states,
rounded-full pills, Material/Ant icons (use Lucide), or saturated background
colours. Numbers are always monospace. Status is always information, never
decoration.
```

---

## 7 · Iteration prompts (per-screen refinements after the first generation)

If a screen doesn't land, send Stitch one of these short follow-ups:

- **Run detail polish:**
  > "Make the three metric cards taller (h-40), put the metric name in micro
  > uppercase tracking-wide, and add a 2-line skeleton state that pulses while
  > the run is still running."

- **Comparison view tightness:**
  > "In the leaderboard, render the best value of each metric column in bold
  > with a dotted underline. Add a tiny coloured square (8px) to the left of
  > each experiment name matching the chart series colour for that row."

- **Login restraint:**
  > "Remove the gradient background. The login page should be the same
  > #09090B as every other page, with the card the only thing breaking the
  > void. Add a single mono line at the bottom: '● Backend connected' or
  > '● Backend unavailable' depending on /health."

- **Empty states:**
  > "Replace all illustration-based empty states with a 1px dashed border
  > card containing a single muted sentence. No icons, no images."

---

## 8 · What to do after Stitch generates

1. Click "Export → React" in Stitch
2. Click "Export → design.md" — save it to `tfg-arquitectura/frontend/design.md`
3. Drop the React export folder into `tfg-arquitectura/frontend/src/components/stitch/`
4. Tell Claude Code: *"Read frontend/CLAUDE.md and frontend/design.md, translate the
   tokens into src/index.css + tailwind.config.js, then reskin the six pages keeping
   all TanStack Query hooks intact."*
