import { Moon, Sun, Sparkles } from "lucide-react";
import { useTheme, type Theme } from "@/lib/theme";
import { cn } from "@/lib/utils";

/* Compact icon button that cycles dark → light → violet → dark.
 * Reused in the landing nav, the inner TopNav, and Login. */

const META: Record<Theme, { icon: typeof Moon; label: string; next: string }> = {
  dark:   { icon: Moon,     label: "Dark theme",   next: "Switch to light" },
  light:  { icon: Sun,      label: "Light theme",  next: "Switch to violet" },
  violet: { icon: Sparkles, label: "Violet theme", next: "Switch to dark" },
};

export function ThemeToggle({ className = "" }: { className?: string }) {
  const { theme, cycleTheme } = useTheme();
  const { icon: Icon, label, next } = META[theme];

  return (
    <button
      type="button"
      onClick={cycleTheme}
      title={`${label} — ${next}`}
      aria-label={`${label}. ${next}.`}
      className={cn(
        "flex items-center justify-center h-8 w-8 rounded border border-border",
        "text-foreground-muted hover:text-foreground hover:border-border-strong",
        "transition-colors",
        theme === "violet" && "text-mcp",
        className,
      )}
    >
      <Icon size={15} />
    </button>
  );
}

export default ThemeToggle;
