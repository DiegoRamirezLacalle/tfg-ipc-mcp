---
name: Terminal Analytical
colors:
  surface: '#15121b'
  surface-dim: '#15121b'
  surface-bright: '#3b3742'
  surface-container-lowest: '#0f0d15'
  surface-container-low: '#1d1a23'
  surface-container: '#211e27'
  surface-container-high: '#2c2832'
  surface-container-highest: '#37333d'
  on-surface: '#e7e0ed'
  on-surface-variant: '#cbc3d7'
  inverse-surface: '#e7e0ed'
  inverse-on-surface: '#322f39'
  outline: '#958ea0'
  outline-variant: '#494454'
  surface-tint: '#d0bcff'
  primary: '#d0bcff'
  on-primary: '#3c0091'
  primary-container: '#a078ff'
  on-primary-container: '#340080'
  inverse-primary: '#6d3bd7'
  secondary: '#4cd7f6'
  on-secondary: '#003640'
  secondary-container: '#03b5d3'
  on-secondary-container: '#00424e'
  tertiary: '#ffb869'
  on-tertiary: '#482900'
  tertiary-container: '#ca801e'
  on-tertiary-container: '#3f2300'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e9ddff'
  primary-fixed-dim: '#d0bcff'
  on-primary-fixed: '#23005c'
  on-primary-fixed-variant: '#5516be'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#ffdcbb'
  tertiary-fixed-dim: '#ffb869'
  on-tertiary-fixed: '#2c1700'
  on-tertiary-fixed-variant: '#673d00'
  background: '#15121b'
  on-background: '#e7e0ed'
  surface-variant: '#37333d'
typography:
  display-lg:
    fontFamily: Geist Sans
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.1'
    letterSpacing: -0.05em
  headline-md:
    fontFamily: Geist Sans
    fontSize: 20px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  body-base:
    fontFamily: Geist Sans
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: 0em
  body-sm:
    fontFamily: Geist Sans
    fontSize: 12px
    fontWeight: '400'
    lineHeight: '1.4'
    letterSpacing: 0em
  data-lg:
    fontFamily: Geist Mono
    fontSize: 18px
    fontWeight: '500'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  data-base:
    fontFamily: Geist Mono
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.4'
    letterSpacing: -0.01em
  data-sm:
    fontFamily: Geist Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: '1.2'
    letterSpacing: 0em
  label-caps:
    fontFamily: Geist Mono
    fontSize: 11px
    fontWeight: '600'
    lineHeight: '1'
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  '4': 4px
  '8': 8px
  '12': 12px
  '16': 16px
  '24': 24px
  '32': 32px
  '48': 48px
  '64': 64px
  nav-height: 56px
---

## Brand & Style

The design system is engineered for high-stakes economic forecasting and data analysis. It adopts a "Bloomberg Terminal meets academic journal" aesthetic, prioritizing information density and legibility over decorative elements. The visual language is austere, technical, and authoritative, evoking the precision of a command-line interface refined for professional financial research.

The style is a hybrid of **Minimalism** and **Brutalism**, utilizing strict grid alignments, clear borders, and monospace-heavy layouts. There are no gradients, blurs, or glow effects; depth is communicated solely through tonal shifts and structural framing. The emotional response should be one of intense focus, objectivity, and expert control.

## Colors

The color palette is optimized for long-duration viewing in dark environments. The foundation is built on deep zinc-blacks and charcoal grays, providing a neutral stage for data visualization. 

- **Background & Surfaces:** A tiered approach using `#09090B` for the base and `#0F0F12` for containers.
- **Accents:** The MCP accent (`#8B5CF6` violet) is used sparingly for primary actions and active states to maintain focus.
- **Data Status:** Semantic colors (Success, Info, Warning, Destructive) follow standard financial patterns but are adjusted for high contrast against the dark background.
- **Chart Series:** A distinct 6-color palette designed for maximum differentiation in complex multi-line forecasting charts.

## Typography

Typography is the primary driver of hierarchy in this design system. We utilize two distinct typefaces:

1.  **Geist Sans:** Used for general UI, prose, and descriptive headers. Headlines utilize tight tracking (`-0.05em` to `-0.02em`) to create a dense, editorial look.
2.  **Geist Mono:** Used for all quantitative data, timestamps, identifiers, and labels. This ensures that columns of numbers align vertically for easier comparison.

Large displays (>32px) should be avoided to maximize space for data. "Data-base" is the workhorse size for all table entries and forecasting inputs.

## Layout & Spacing

The layout philosophy is based on a **high-density fluid grid**. The interface is designed to fill the viewport, minimizing scrolling in favor of "at-a-glance" dashboarding.

- **Grid:** A 12-column system with tight 12px gutters. Elements should align strictly to the 4px baseline grid.
- **Navigation:** A fixed 56px top bar contains global navigation and the monospace brand identifier "tfg-ipc-mcp".
- **Density:** Padding within containers is kept to a minimum (usually 8px or 12px) to maximize the "data per square inch" ratio.
- **Breakpoints:** On desktop, use side-by-side comparison panes. On mobile, reflow into a single column with horizontal scrolling enabled for large data tables.

## Elevation & Depth

This design system eschews shadows in favor of **structural borders and tonal layers**. 

1.  **Level 0 (Background):** `#09090B` - The canvas for the entire platform.
2.  **Level 1 (Card/Section):** `#0F0F12` - Used for primary content areas, defined by a 1px border of `#27272A`.
3.  **Level 2 (Active/Hover/Floating):** `#18181B` - Used for hover states on list items, selected menu options, or utility bars.

Depth is communicated through the "stacking" of these monochromatic grays. Use 1px solid borders for all container separations to maintain a crisp, technical feel.

## Shapes

The shape language is rigid and geometric. Rounded corners are used only to prevent the UI from feeling "sharp," but are never aggressive enough to look friendly or playful.

- **4px (sm):** Used for inputs, checkboxes, and small utility buttons.
- **6px (md):** Default for cards, modules, and primary interface buttons.
- **8px (lg):** Reserved for large layout containers or modals.

Pills and full-round elements are strictly prohibited. All buttons and tags must remain rectangular with the specified 4px or 6px radii.

## Components

- **Buttons:** Low-profile and flat. Primary buttons use the Violet accent with white text. Secondary buttons use a border-only "ghost" style. No shadows or gradients.
- **Data Tables:** The most critical component. Features 1px borders between rows, Geist Mono for all cell data, and a slightly darker header background (`#18181B`). Hover states use `#18181B`.
- **Inputs:** Dark backgrounds (`#09090B`) with `#27272A` borders. Use Geist Mono for text entry to align with data visualization. Active state uses a 1px violet border.
- **Chips/Status:** Small, rectangular tags with 4px radii. Use semantic colors for text/border with a low-opacity version for the background.
- **Charts:** Flat line and bar charts using the Chart Palette. No area fills unless highly transparent (5-10% opacity). Use Lucide icons (thin or regular weight) for all UI signifiers.
- **Navigation Bar:** Fixed 56px height. Background `#09090B` with a solid bottom border. Monospace brand text "tfg-ipc-mcp" should be lowercase.