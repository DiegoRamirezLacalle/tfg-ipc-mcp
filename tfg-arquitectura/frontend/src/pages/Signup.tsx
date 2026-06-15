import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "motion/react";
import { useSignup } from "@/lib/queries";

export default function Signup() {
  const navigate = useNavigate();
  const signup = useSignup();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    signup.mutate({ email, password }, { onSuccess: () => navigate("/experiments") });
  };

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center px-4 gap-6">
      <div className="absolute top-0 left-0 right-0 gold-rule" />

      <motion.div className="text-center" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <Link to="/" className="font-mono text-data-lg lowercase tracking-tighter text-foreground hover:text-gold transition-colors">
          tfg-ipc-mcp<span className="cursor-block" />
        </Link>
      </motion.div>

      <motion.form
        onSubmit={onSubmit}
        className="w-full max-w-[360px] bg-card border border-border rounded p-8 flex flex-col gap-6 relative overflow-hidden"
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-gold/40 to-transparent" />

        <div className="flex flex-col gap-1">
          <h2 className="font-mono text-data-lg uppercase tracking-tight text-foreground">Create Account</h2>
          <p className="font-mono text-data-sm text-foreground-muted">
            Request researcher access to the platform.
          </p>
        </div>

        <div className="flex flex-col gap-4">
          <label className="flex flex-col gap-1.5">
            <span className="micro">Email</span>
            <input
              type="email" autoComplete="email" required
              value={email} onChange={(e) => setEmail(e.target.value)}
              className="h-10 px-3 bg-background border border-border rounded font-mono text-data-base text-foreground focus:border-gold focus:ring-1 focus:ring-gold/20 outline-none transition-colors placeholder:text-foreground-subtle"
              placeholder="you@institution.edu"
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="micro">Password</span>
            <input
              type="password" autoComplete="new-password" required minLength={8}
              value={password} onChange={(e) => setPassword(e.target.value)}
              className="h-10 px-3 bg-background border border-border rounded font-mono text-data-base text-foreground focus:border-gold focus:ring-1 focus:ring-gold/20 outline-none transition-colors placeholder:text-foreground-subtle"
              placeholder="min. 8 characters"
            />
          </label>
        </div>

        {signup.error && (
          <p className="font-mono text-data-sm text-destructive">{(signup.error as Error).message}</p>
        )}

        <button
          type="submit" disabled={signup.isPending}
          className="h-10 bg-gold/90 hover:bg-gold text-background font-mono text-label-caps uppercase tracking-widest rounded transition-colors disabled:opacity-50"
        >
          {signup.isPending ? "Creating..." : "Create account ->"}
        </button>

        <p className="text-center font-mono text-data-sm text-foreground-muted">
          Already have access?{" "}
          <Link to="/login" className="text-foreground hover:text-gold transition-colors">
            Log in
          </Link>
        </p>
      </motion.form>

      <div className="absolute bottom-0 left-0 right-0 gold-rule" />
    </div>
  );
}
