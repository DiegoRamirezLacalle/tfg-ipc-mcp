import { useId } from "react";
import { motion } from "motion/react";

/* ────────────────────────────────────────────────────────────────────────
 * ForecastDivider — a quiet callback to the hero's forecast-line motif.
 *
 * A single thin trajectory that gently undulates across a full-width strip,
 * with the gold convergence dot on the right. Far calmer than the hero —
 * it's connective tissue between sections, not a second spectacle. Used in
 * place of the plain gold rule on the home page.
 * ──────────────────────────────────────────────────────────────────────── */

export function ForecastDivider({
  className = "",
  theme = "dark",
}: { className?: string; theme?: "dark" | "light" | "violet" }) {
  const uid = useId().replace(/:/g, "");
  const isLight = theme === "light";
  const violet = isLight ? "#6D28D9" : "#8B5CF6";
  const gold = isLight ? "#A8741F" : "#E0B96A";

  return (
    <div className={"relative h-16 w-full overflow-hidden " + className} aria-hidden="true">
      <svg
        viewBox="0 0 1200 80"
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full"
      >
        <defs>
          <linearGradient id={`fade-${uid}`} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={violet} stopOpacity="0" />
            <stop offset="35%" stopColor={violet} stopOpacity={isLight ? 0.7 : 0.5} />
            <stop offset="70%" stopColor={gold} stopOpacity={isLight ? 0.7 : 0.5} />
            <stop offset="100%" stopColor={gold} stopOpacity="0" />
          </linearGradient>
        </defs>

        <motion.path
          fill="none"
          stroke={`url(#fade-${uid})`}
          strokeWidth="1.5"
          initial={{ d: "M 0 42 C 300 30 500 54 750 40 C 950 30 1080 46 1200 40" }}
          animate={{
            d: [
              "M 0 42 C 300 30 500 54 750 40 C 950 30 1080 46 1200 40",
              "M 0 38 C 300 52 500 32 750 46 C 950 52 1080 36 1200 42",
              "M 0 42 C 300 30 500 54 750 40 C 950 30 1080 46 1200 40",
            ],
          }}
          transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
        />
      </svg>
      {/* gold convergence dot, pulsing softly */}
      <motion.span
        className="absolute top-1/2 right-[3%] h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-gold"
        style={{ boxShadow: "0 0 10px hsl(var(--gold) / 0.8)" }}
        animate={{ opacity: [0.3, 1, 0.3] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}

export default ForecastDivider;
