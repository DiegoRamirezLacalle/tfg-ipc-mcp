import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "motion/react";
import { ArrowRight, FlaskConical, BarChart3, SlidersHorizontal, Radio, ChevronDown } from "lucide-react";

import { NewsPulse } from "@/components/news/NewsPulse";
import { FlowField } from "@/components/visuals/FlowField";
import { AnimatedField } from "@/components/visuals/AnimatedField";
import { ForecastDivider } from "@/components/visuals/ForecastDivider";
import { ThemeToggle } from "@/components/app/ThemeToggle";
import { useTheme } from "@/lib/theme";

const DOORS = [
  {
    n: "01",
    icon: FlaskConical,
    title: "The Contest",
    rq: "Classical vs foundation",
    desc: "Run ARIMA, SARIMA and Ridge against TimesFM, Chronos-2 and TimeGPT — with or without MCP context — on a rolling backtest.",
    to: "/experiments",
  },
  {
    n: "02",
    icon: BarChart3,
    title: "The Evidence",
    rq: "RQ1 · who wins, and is it real?",
    desc: "Leaderboard, skill score vs the seasonal naïve, horizon-wise error decay, and Diebold-Mariano significance tests.",
    to: "/compare",
  },
  {
    n: "03",
    icon: SlidersHorizontal,
    title: "The Mechanism",
    rq: "RQ2 · do signals add value?",
    desc: "Perturb ECB rates and Fed/US-CPI signals and watch the forecast respond — the marginal effect of MCP context, made tangible.",
    to: "/simulator",
  },
];

const MODELS = [
  { name: "Naïve", color: "#E4E4E7" },
  { name: "SARIMA", color: "#06B6D4" },
  { name: "Ridge", color: "#F59E0B" },
  { name: "TimesFM", color: "#8B5CF6" },
  { name: "Chronos-2", color: "#10B981" },
  { name: "TimeGPT", color: "#F43F5E" },
];

function reveal(delay = 0) {
  return {
    initial:     { opacity: 0, y: 22 },
    whileInView: { opacity: 1, y: 0 },
    viewport:    { once: true, margin: "-60px" },
    transition:  { duration: 0.65, delay, ease: [0.16, 1, 0.3, 1] as [number, number, number, number] },
  };
}

export default function Landing() {
  const [scrolled, setScrolled] = useState(false);
  const { theme } = useTheme();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-hidden">
      {/* -- NAV ------------------------------------------- */}
      <nav
        className={
          "fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 h-14 transition-colors duration-500 " +
          (scrolled
            ? "border-b border-border/60 bg-background/80 backdrop-blur-md"
            : "border-b border-transparent bg-transparent")
        }
      >
        <span className="font-mono text-data-lg lowercase tracking-tighter text-foreground">
          tfg-ipc-mcp<span className="cursor-block" />
        </span>
        <div className="flex items-center gap-3">
          <ThemeToggle />
          <Link to="/login" className="font-mono text-data-sm text-foreground-muted hover:text-foreground transition-colors px-2 py-1">
            Log in
          </Link>
          <Link to="/signup" className="font-mono text-data-sm text-foreground-muted hover:text-foreground transition-colors px-2 py-1">
            Sign up
          </Link>
          <Link to="/experiments" className="flex items-center gap-1.5 font-mono text-data-sm bg-mcp hover:bg-mcp/90 text-white px-4 py-1.5 rounded transition-colors">
            Enter platform <ArrowRight size={11} />
          </Link>
        </div>
      </nav>

      {/* -- HERO ------------------------------------------ */}
      <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
        {/* Layered living background (z-0, content sits at z-10 above):
            1. WebGL volumetric nebula — cinematic violet/gold/indigo fog
            2. animated SVG forecast lines drawing on top (guaranteed visible)
            3. readability scrim under the headline + bottom fade            */}
        <div className="absolute inset-0 z-0 pointer-events-none">
          <FlowField className="absolute inset-0 h-full w-full" speed={0.5} interactive theme={theme} />
          <AnimatedField
            className={"absolute inset-0 h-full w-full " + (theme === "light" ? "mix-blend-multiply" : "mix-blend-screen")}
            theme={theme}
          />
          {/* scrim under the headline so it always reads (light vs dark) */}
          <div className="hero-scrim absolute inset-0" />
          <div className="absolute bottom-0 left-0 right-0 h-56 bg-gradient-to-t from-background via-background/70 to-transparent" />
        </div>

        <div className="relative z-10 flex flex-col items-center text-center gap-7 px-6 max-w-3xl">
          <motion.p className="micro uppercase tracking-widest text-gold"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
          >
            TFG · Inflation Forecasting · Model Context Protocol
          </motion.p>

          <motion.h1 className="font-grotesk font-bold text-gradient-hero text-[clamp(2.8rem,8.5vw,6rem)] leading-[0.98] tracking-[-0.03em]"
            initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35, duration: 0.85, ease: [0.16, 1, 0.3, 1] }}
          >
            Forecasting the<br />
            Price of Tomorrow.
          </motion.h1>

          <motion.p className="font-mono text-data-base text-foreground-muted max-w-xl leading-relaxed"
            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.55, duration: 0.7 }}
          >
            An interactive test bench for the thesis: can foundation time-series models, enriched
            with MCP macro context, beat classical baselines at forecasting inflation?
          </motion.p>

          {/* The two research questions — the spine of everything */}
          <motion.div className="flex flex-col sm:flex-row gap-3 w-full max-w-2xl"
            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }}
          >
            {[
              ["RQ1", "Do foundation models beat classical baselines?"],
              ["RQ2", "Do MCP exogenous signals add predictive value?"],
            ].map(([tag, q]) => (
              <div key={tag} className="flex-1 card-glass rounded p-3 flex items-center gap-3 text-left">
                <span className="font-mono text-[10px] text-gold border border-gold/40 rounded px-1.5 py-0.5 shrink-0">{tag}</span>
                <span className="font-mono text-data-sm text-foreground-muted leading-snug">{q}</span>
              </div>
            ))}
          </motion.div>

          <motion.div className="flex items-center gap-3"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9 }}
          >
            <Link to="/experiments" className="flex items-center gap-2 h-10 px-7 bg-mcp hover:bg-mcp/90 text-white font-mono text-label-caps uppercase tracking-widest rounded transition-colors">
              Enter platform <ArrowRight size={12} />
            </Link>
            <Link to="/today" className="flex items-center gap-2 h-10 px-5 card-glass rounded text-foreground-muted hover:text-foreground font-mono text-data-sm transition-colors">
              <Radio size={12} className="text-gold" /> Today's pulse
            </Link>
          </motion.div>
        </div>

        {/* Scroll cue */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center gap-1"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.4, duration: 0.8 }}
        >
          <span className="micro uppercase text-foreground-subtle">Scroll</span>
          <ChevronDown size={16} className="text-foreground-subtle animate-scroll-cue" />
        </motion.div>
      </section>

      {/* -- MODELS STRIP ---------------------------------- */}
      <div className="relative border-y border-border bg-muted/30 backdrop-blur-sm py-3 px-6">
        <div className="max-w-5xl mx-auto flex flex-wrap items-center justify-center gap-x-8 gap-y-2">
          <span className="micro uppercase text-foreground-subtle">On the leaderboard</span>
          {MODELS.map((m) => (
            <span key={m.name} className="flex items-center gap-2 font-mono text-data-sm text-foreground-muted">
              <span className="w-2 h-2 rounded-full" style={{ background: m.color, boxShadow: `0 0 8px ${m.color}66` }} />
              {m.name}
            </span>
          ))}
        </div>
      </div>

      {/* -- THREE DOORS ----------------------------------- */}
      <section className="max-w-5xl mx-auto px-6 py-24 flex flex-col gap-12">
        <motion.div className="section-glow flex flex-col gap-3" {...reveal()}>
          <p className="micro text-foreground-subtle uppercase">Explore the thesis</p>
          <h2 className="font-grotesk font-semibold text-display-lg leading-tight tracking-[-0.02em]">Three ways in.</h2>
          <p className="font-mono text-data-base text-foreground-muted max-w-lg leading-relaxed">
            Each door answers one piece of the argument — run the models, weigh the evidence, probe the mechanism.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {DOORS.map((d, i) => (
            <motion.div key={d.n} {...reveal(i * 0.08)}>
              <Link
                to={d.to}
                className="card-glass glow-mcp rounded-lg p-6 flex flex-col gap-4 h-full group"
              >
                <div className="flex items-center justify-between">
                  <d.icon size={20} className="text-gold" />
                  <span className="font-mono text-data-lg text-foreground-subtle group-hover:text-foreground transition-colors">{d.n}</span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="font-sans text-headline-md text-foreground">{d.title}</span>
                  <span className="font-mono text-[10px] text-mcp uppercase tracking-wider">{d.rq}</span>
                </div>
                <p className="font-mono text-data-sm text-foreground-muted leading-relaxed flex-1">{d.desc}</p>
                <span className="flex items-center gap-1.5 font-mono text-data-sm text-foreground-muted group-hover:text-gold transition-colors">
                  Open <ArrowRight size={12} className="group-hover:translate-x-1 transition-transform" />
                </span>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      {/* -- DIVIDER — faint forecast-line callback -------- */}
      <ForecastDivider className="max-w-5xl mx-auto px-6" theme={theme} />

      {/* -- LIVE PULSE ------------------------------------ */}
      <section className="max-w-5xl mx-auto px-6 py-24 flex flex-col gap-8">
        <motion.div className="flex items-end justify-between gap-4 flex-wrap" {...reveal()}>
          <div className="section-glow flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <Radio size={13} className="text-gold" />
              <p className="micro text-gold uppercase">Live · right now</p>
            </div>
            <h2 className="font-grotesk font-semibold text-display-lg leading-tight tracking-[-0.02em]">The economy, as it speaks.</h2>
            <p className="font-mono text-data-base text-foreground-muted max-w-lg leading-relaxed">
              Real inflation headlines from GDELT, scored live by FinBERT through the MCP server —
              the exact sentiment channel the models consume as C1 context.
            </p>
          </div>
          <Link to="/today" className="flex items-center gap-1.5 font-mono text-data-sm text-foreground-muted hover:text-gold transition-colors shrink-0">
            Full pulse <ArrowRight size={12} />
          </Link>
        </motion.div>

        <motion.div {...reveal(0.15)}>
          <NewsPulse limit={6} showRefresh />
        </motion.div>
      </section>

      {/* -- FINAL CTA ------------------------------------- */}
      <section className="relative max-w-5xl mx-auto px-6 py-28 flex flex-col items-center text-center gap-8">
        <motion.div className="section-glow flex flex-col gap-4 items-center" {...reveal()}>
          <h2 className="font-grotesk font-semibold text-display-lg leading-tight tracking-[-0.02em]">
            Ready to put the models<br />
            <span className="text-gradient-hero">to the test?</span>
          </h2>
          <p className="font-mono text-data-base text-foreground-muted max-w-md">
            Run experiments, weigh them with statistical significance, and probe how MCP signals
            move the forecast — all in one place.
          </p>
        </motion.div>
        <motion.div className="flex items-center gap-3" {...reveal(0.2)}>
          <Link to="/experiments" className="flex items-center gap-2 h-11 px-8 bg-mcp hover:bg-mcp/90 text-white font-mono text-label-caps uppercase tracking-widest rounded transition-colors">
            Enter platform <ArrowRight size={13} />
          </Link>
          <Link to="/login" className="h-11 px-6 card-glass rounded text-foreground-muted hover:text-foreground font-mono text-data-sm transition-colors flex items-center">
            Log in
          </Link>
        </motion.div>
      </section>

      {/* -- FOOTER ---------------------------------------- */}
      <footer className="border-t border-border px-6 py-5 flex items-center justify-between">
        <span className="font-mono text-data-sm text-foreground-subtle">tfg-ipc-mcp · 2026</span>
        <span className="font-mono text-data-sm text-foreground-subtle">Diego Ramírez · TFG</span>
      </footer>
    </div>
  );
}
