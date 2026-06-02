import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "motion/react";
import { useLogin } from "@/lib/queries";

const LINES = [
  { d: "M 0 180 C 200 190 400 210 700 230 C 900 240 1100 248 1200 250", stroke: "#E4E4E7", opacity: 0.5, delay: 0 },
  { d: "M 0 140 C 150 165 350 195 650 225 C 850 238 1050 247 1200 250", stroke: "#06B6D4", opacity: 0.55, delay: 0.25 },
  { d: "M 0 220 C 200 225 400 238 700 248 C 900 250 1100 250 1200 250", stroke: "#F59E0B", opacity: 0.5, delay: 0.4 },
  { d: "M 0 300 C 200 275 400 260 700 255 C 900 252 1100 251 1200 250", stroke: "#8B5CF6", opacity: 0.6, delay: 0.55 },
  { d: "M 0 360 C 200 320 400 290 700 265 C 900 255 1100 251 1200 250", stroke: "#10B981", opacity: 0.5, delay: 0.7 },
  { d: "M 0 410 C 200 355 400 310 700 272 C 900 258 1100 252 1200 250", stroke: "#F43F5E", opacity: 0.45, delay: 0.85 },
];

function HeroChart() {
  return (
    <svg
      viewBox="0 0 1200 500"
      preserveAspectRatio="xMidYMid slice"
      className="absolute inset-0 w-full h-full"
    >
      {/* Grid */}
      {[100, 200, 300, 400].map((y) => (
        <line key={y} x1="0" y1={y} x2="1200" y2={y} stroke="#27272A" strokeWidth="1" />
      ))}
      {[200, 400, 600, 800, 1000].map((x) => (
        <line key={x} x1={x} y1="0" x2={x} y2="500" stroke="#27272A" strokeWidth="1" />
      ))}

      {/* Convergence target dot */}
      <motion.circle
        cx={1200}
        cy={250}
        r={4}
        fill="#E0B96A"
        initial={{ opacity: 0, scale: 0 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 1.8, duration: 0.4 }}
      />

      {/* Forecast lines */}
      {LINES.map((line, i) => (
        <motion.path
          key={i}
          d={line.d}
          stroke={line.stroke}
          strokeWidth={1.5}
          fill="none"
          opacity={line.opacity}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 2.2, delay: line.delay, ease: [0.16, 1, 0.3, 1] }}
        />
      ))}

      {/* "IPC" label */}
      <motion.text
        x={24}
        y={248}
        fill="#71717A"
        fontFamily="'Geist Mono', monospace"
        fontSize={10}
        letterSpacing="0.1em"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2, duration: 0.6 }}
      >
        IPC · YoY%
      </motion.text>
    </svg>
  );
}

export default function Login() {
  const navigate = useNavigate();
  const login = useLogin();
  const [email, setEmail] = useState("admin@tfg.local");
  const [password, setPassword] = useState("changeme-dev");

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    login.mutate({ email, password }, { onSuccess: () => navigate("/experiments") });
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      {/* Animated chart background */}
      <div className="absolute inset-0 pointer-events-none">
        <HeroChart />
      </div>

      {/* Top gold rule */}
      <div className="absolute top-0 left-0 right-0 gold-rule" />

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center px-4 gap-8">
        {/* Wordmark */}
        <motion.div
          className="text-center"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        >
          <span className="font-mono text-data-lg lowercase tracking-tighter text-foreground">
            tfg-ipc-mcp<span className="cursor-block" />
          </span>
        </motion.div>

        {/* Keynes quote */}
        <motion.blockquote
          className="text-center max-w-xs"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.9 }}
        >
          <p className="font-display text-base italic text-foreground-subtle leading-relaxed">
            "The difficulty lies not so much in developing new ideas as in escaping from old ones."
          </p>
          <cite className="font-mono text-label-caps text-foreground-subtle/50 uppercase tracking-widest mt-2 block not-italic">
            — J.M. Keynes
          </cite>
        </motion.blockquote>

        {/* Login card */}
        <motion.form
          onSubmit={onSubmit}
          className="w-full max-w-[360px] bg-card border border-border rounded p-8 flex flex-col gap-6 relative overflow-hidden"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-mcp/40 to-transparent" />

          <div className="flex flex-col gap-1">
            <h2 className="font-mono text-data-lg uppercase tracking-tight text-foreground">System Access</h2>
            <p className="font-mono text-data-sm text-foreground-muted">
              Secure authentication for inflation forecasting tools.
            </p>
          </div>

          <div className="flex flex-col gap-4">
            <label className="flex flex-col gap-1.5">
              <span className="micro">Email</span>
              <input
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="h-10 px-3 bg-background border border-border rounded font-mono text-data-base text-foreground focus:border-mcp focus:ring-1 focus:ring-mcp/20 outline-none transition-colors placeholder:text-foreground-subtle"
                placeholder="analyst@tfg.com"
              />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="micro">Password</span>
              <input
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="h-10 px-3 bg-background border border-border rounded font-mono text-data-base text-foreground focus:border-mcp focus:ring-1 focus:ring-mcp/20 outline-none transition-colors placeholder:text-foreground-subtle"
                placeholder="••••••••"
              />
            </label>
          </div>

          {login.error && (
            <p className="font-mono text-data-sm text-destructive">
              {(login.error as Error).message}
            </p>
          )}

          <button
            type="submit"
            disabled={login.isPending}
            className="h-10 bg-mcp hover:bg-mcp/90 text-white font-mono text-label-caps uppercase tracking-widest rounded transition-colors disabled:opacity-50"
          >
            {login.isPending ? "Authenticating…" : "Authenticate →"}
          </button>

          <p className="text-center font-mono text-label-caps text-foreground-subtle/40 uppercase tracking-widest">
            Confidential &amp; Proprietary
          </p>
        </motion.form>
      </div>

      {/* Bottom gold rule */}
      <div className="absolute bottom-0 left-0 right-0 gold-rule" />
    </div>
  );
}
