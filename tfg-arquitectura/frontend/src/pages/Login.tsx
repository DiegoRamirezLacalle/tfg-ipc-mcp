import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "motion/react";
import { useLogin } from "@/lib/queries";
import { FlowField } from "@/components/visuals/FlowField";
import { AnimatedField } from "@/components/visuals/AnimatedField";
import { ThemeToggle } from "@/components/app/ThemeToggle";
import { useTheme } from "@/lib/theme";

export default function Login() {
  const navigate = useNavigate();
  const login = useLogin();
  const { theme } = useTheme();
  const [email, setEmail] = useState("admin@tfg.local");
  const [password, setPassword] = useState("changeme-dev");

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    login.mutate({ email, password }, { onSuccess: () => navigate("/experiments") });
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      {/* Living background — a dim, slow echo of the landing hero */}
      <div className="absolute inset-0 pointer-events-none">
        <AnimatedField
          className={"absolute inset-0 h-full w-full opacity-40 " + (theme === "light" ? "mix-blend-multiply" : "")}
          theme={theme}
        />
        <FlowField
          className={"absolute inset-0 h-full w-full opacity-40 " + (theme === "light" ? "" : "mix-blend-screen")}
          speed={0.15}
          theme={theme}
        />
        <div className="absolute inset-0 bg-background/40" />
      </div>

      {/* Theme toggle */}
      <div className="absolute top-4 right-4 z-20">
        <ThemeToggle />
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
