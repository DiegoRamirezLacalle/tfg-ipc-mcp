/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "1.5rem",
      screens: { "2xl": "1440px" },
    },
    extend: {
      colors: {
        // shadcn-style semantic tokens (driven by CSS vars in index.css)
        border: "hsl(var(--border))",
        "border-strong": "hsl(var(--border-strong))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        "foreground-muted": "hsl(var(--foreground-muted))",
        "foreground-subtle": "hsl(var(--foreground-subtle))",
        card: "hsl(var(--card))",
        muted: "hsl(var(--muted))",
        // status
        success: "hsl(var(--success))",
        info: "hsl(var(--info))",
        warning: "hsl(var(--warning))",
        destructive: "hsl(var(--destructive))",
        // brand accents
        mcp: "hsl(var(--mcp))", // violet — RESERVED for MCP-related UI only
        gold: "hsl(var(--gold))", // champagne gold — thesis identity accent
        // chart series (one per model slug — used in Recharts <Line stroke> etc.)
        series: {
          naive:   "hsl(var(--series-naive))",
          sarima:  "hsl(var(--series-sarima))",
          ridge:   "hsl(var(--series-ridge))",
          timesfm: "hsl(var(--series-timesfm))",
          chronos: "hsl(var(--series-chronos))",
          timegpt: "hsl(var(--series-timegpt))",
        },
      },
      borderRadius: {
        sm: "3px",
        DEFAULT: "4px",
        md: "6px",
        lg: "8px",
      },
      fontFamily: {
        sans:    ["Geist", "ui-sans-serif", "system-ui", "sans-serif"],
        mono:    ["Geist Mono", "ui-monospace", "monospace"],
        display: ['"Instrument Serif"', "ui-serif", "Georgia", "serif"],
      },
      fontSize: {
        // Stitch typography scale
        "display-xl":  ["56px", { lineHeight: "1.05", letterSpacing: "-0.04em", fontWeight: "400" }],
        "display-lg":  ["32px", { lineHeight: "1.1",  letterSpacing: "-0.03em", fontWeight: "600" }],
        "headline-md": ["20px", { lineHeight: "1.2",  letterSpacing: "-0.02em", fontWeight: "600" }],
        "body-base":   ["14px", { lineHeight: "1.5",  letterSpacing: "0em",     fontWeight: "400" }],
        "body-sm":     ["12px", { lineHeight: "1.4",  letterSpacing: "0em",     fontWeight: "400" }],
        "data-lg":     ["18px", { lineHeight: "1.2",  letterSpacing: "-0.02em", fontWeight: "500" }],
        "data-base":   ["14px", { lineHeight: "1.4",  letterSpacing: "-0.01em", fontWeight: "400" }],
        "data-sm":     ["12px", { lineHeight: "1.2",  letterSpacing: "0em",     fontWeight: "400" }],
        "data-2xl":    ["32px", { lineHeight: "1.0",  letterSpacing: "-0.04em", fontWeight: "500" }],
        "label-caps":  ["10px", { lineHeight: "1",    letterSpacing: "0.12em",  fontWeight: "600" }],
      },
      spacing: {
        "nav-height": "56px",
      },
      keyframes: {
        "fade-up": {
          "0%":   { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "draw-line": {
          "0%":   { strokeDashoffset: "1000" },
          "100%": { strokeDashoffset: "0" },
        },
        "blink": {
          "0%, 49%":   { opacity: "1" },
          "50%, 100%": { opacity: "0" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "0.6" },
          "50%":      { opacity: "1" },
        },
        "marquee": {
          "0%":   { transform: "translateX(0)" },
          "100%": { transform: "translateX(-33.333%)" },
        },
      },
      animation: {
        "fade-up":    "fade-up 600ms cubic-bezier(0.16, 1, 0.3, 1) both",
        "draw-line":  "draw-line 2.5s ease-out forwards",
        "blink":      "blink 1.1s steps(1) infinite",
        "pulse-soft": "pulse-soft 2.4s ease-in-out infinite",
        "marquee":    "marquee 32s linear infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
