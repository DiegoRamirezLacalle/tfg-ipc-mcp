import { useEffect, useState } from "react";

/* ------------------------------------------------------------------------
 * Theme - three palettes switched by a class on <html>.
 *   dark   -> zinc near-black (default)
 *   light  -> white / near-white
 *   violet -> violet-tinted dark (branded)
 *
 * Mirrors the useAuth pattern (src/lib/auth.ts): a hook backed by
 * localStorage + a `storage` listener for cross-tab sync. No context.
 * Token values per theme live in src/index.css under .theme-* classes.
 * ------------------------------------------------------------------------ */

export type Theme = "dark" | "light" | "violet";

export const THEMES: Theme[] = ["dark", "light", "violet"];
const STORAGE_KEY = "tfg_theme";

export function readTheme(): Theme {
  const raw = localStorage.getItem(STORAGE_KEY);
  return raw === "light" || raw === "violet" || raw === "dark" ? raw : "dark";
}

/** Set the <html> class. `dark`/`violet` also carry the `dark` class so any
 *  Tailwind `dark:` variants keep working; `light` drops it. */
export function applyTheme(theme: Theme) {
  const el = document.documentElement;
  el.classList.remove("theme-dark", "theme-light", "theme-violet", "dark", "light");
  el.classList.add(`theme-${theme}`);
  el.classList.add(theme === "light" ? "light" : "dark");
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => readTheme());

  // Apply on mount + whenever it changes.
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  // Keep tabs in sync (same approach as useAuth).
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setThemeState(readTheme());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const setTheme = (t: Theme) => {
    localStorage.setItem(STORAGE_KEY, t);
    setThemeState(t);
  };

  const cycleTheme = () => {
    const next = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
    setTheme(next);
  };

  return { theme, setTheme, cycleTheme };
}
