import { useId } from "react";
import { motion } from "motion/react";

/* ────────────────────────────────────────────────────────────────────────
 * AnimatedField — dramatic, looping "forecast being computed" lines.
 *
 * Each trajectory DRAWS itself left→right (Framer `pathLength` 0→1) with a
 * glowing leading dot that races along the curve (CSS `offset-path`), holds
 * lit for a beat, fades out, then redraws — forever, staggered across lines.
 * Reads as a live prediction being generated. Pure SVG/CSS + Framer Motion,
 * so it can never silently fail; the WebGL nebula glows behind it.
 *
 * A faint terminal grid (the motif from the original landing) sits underneath.
 * ──────────────────────────────────────────────────────────────────────── */

type FieldTheme = "dark" | "light" | "violet";

interface Trace {
  d: string;
  /** stroke colour per theme: [dark/violet, light] */
  colors: [string, string];
  width: number;
  /** seconds for one full draw→hold→fade cycle */
  cycle: number;
  /** start offset so lines fire in sequence, not all at once */
  delay: number;
}

// Paths span x: 0..1200, y around the 250 mid-line — rising/converging curves
// that read like forecast fans homing toward a target on the right.
// Light colours are darker/saturated so they read on a white page.
const TRACES: Trace[] = [
  { d: "M 0 410 C 250 380 480 300 720 250 C 920 210 1080 250 1200 240", colors: ["#8B5CF6", "#6D28D9"], width: 2.5,  cycle: 6.0, delay: 0.0 },
  { d: "M 0 360 C 240 340 470 250 700 230 C 920 215 1080 245 1200 248", colors: ["#06B6D4", "#0E7490"], width: 2.0,  cycle: 6.4, delay: 0.5 },
  { d: "M 0 300 C 250 280 470 240 700 250 C 930 258 1080 240 1200 252", colors: ["#E0B96A", "#A8741F"], width: 2.75, cycle: 5.6, delay: 1.0 },
  { d: "M 0 200 C 240 220 470 270 700 255 C 930 245 1080 250 1200 244", colors: ["#10B981", "#047857"], width: 1.75, cycle: 6.8, delay: 1.6 },
  { d: "M 0 130 C 250 170 480 230 720 248 C 930 260 1080 246 1200 250", colors: ["#F43F5E", "#BE123C"], width: 1.5,  cycle: 7.2, delay: 2.1 },
];

const GRID_Y = [80, 160, 240, 320, 400];
const GRID_X = [150, 350, 550, 750, 950, 1150];

export function AnimatedField({
  className = "",
  theme = "dark",
}: { className?: string; theme?: FieldTheme }) {
  const uid = useId().replace(/:/g, "");
  const isLight = theme === "light";
  const ci = isLight ? 1 : 0; // colour index into Trace.colors
  const gridStroke = isLight ? "#D8D2E6" : "#27272A";
  const dotColor = isLight ? "#A8741F" : "#E0B96A";

  return (
    <svg
      viewBox="0 0 1200 500"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
      className={className}
    >
      <defs>
        <filter id={`soft-${uid}`} x="-20%" y="-60%" width="140%" height="220%">
          <feGaussianBlur stdDeviation="3.5" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* faint terminal grid — the motif from the original landing */}
      <g stroke={gridStroke} strokeWidth="1" opacity="0.6">
        {GRID_Y.map((y) => (
          <line key={`y${y}`} x1="0" y1={y} x2="1200" y2={y} />
        ))}
        {GRID_X.map((x) => (
          <line key={`x${x}`} x1={x} y1="0" x2={x} y2="500" />
        ))}
      </g>

      {/* convergence target — gold pulse on the right (where forecasts home) */}
      <motion.circle
        cx={1150} cy={248} r={5} fill={dotColor}
        animate={{ opacity: [0.3, 1, 0.3], scale: [1, 1.5, 1] }}
        transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
        style={{ transformOrigin: "1150px 248px" }}
      />

      {/* dramatic draw-on forecast trajectories */}
      {TRACES.map((t, i) => (
        <g key={i}>
          {/* the line drawing itself in */}
          <motion.path
            d={t.d}
            fill="none"
            stroke={t.colors[ci]}
            strokeWidth={t.width}
            strokeLinecap="round"
            filter={`url(#soft-${uid})`}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{
              pathLength: [0, 1, 1, 1],
              opacity: [0, 0.95, 0.95, 0],
            }}
            transition={{
              duration: t.cycle,
              times: [0, 0.45, 0.78, 1],
              delay: t.delay,
              repeat: Infinity,
              repeatDelay: 0.4,
              ease: "easeInOut",
            }}
          />
          {/* glowing leading dot that races along the same curve as it draws */}
          <motion.circle
            r={t.width + 2.5}
            fill={t.colors[ci]}
            filter={`url(#soft-${uid})`}
            initial={{ opacity: 0, offsetDistance: "0%" }}
            animate={{
              opacity: [0, 1, 1, 0, 0],
              offsetDistance: ["0%", "0%", "100%", "100%", "100%"],
            }}
            transition={{
              duration: t.cycle,
              times: [0, 0.08, 0.45, 0.55, 1],
              delay: t.delay,
              repeat: Infinity,
              repeatDelay: 0.4,
              ease: "easeInOut",
            }}
            style={{
              offsetPath: `path('${t.d}')`,
              offsetRotate: "0deg",
            }}
          />
        </g>
      ))}
    </svg>
  );
}

export default AnimatedField;
