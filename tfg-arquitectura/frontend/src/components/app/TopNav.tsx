import { NavLink } from "react-router-dom";
import { LogOut } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth";

const NAV = [
  { to: "/experiments",     label: "Experiments", end: true },
  { to: "/experiments/new", label: "New" },
  { to: "/compare",         label: "Compare" },
  { to: "/simulator",       label: "Simulator" },
  { to: "/today",           label: "Today" },
];

export function TopNav() {
  const { user, logout } = useAuth();

  return (
    <header className="fixed top-0 w-full h-[56px] bg-background/95 backdrop-blur-sm border-b border-border z-50 flex items-center justify-between px-6">
      <div className="flex items-center gap-6 h-full">
        <span className="font-mono text-data-lg lowercase tracking-tighter text-foreground select-none">
          tfg-ipc-mcp<span className="cursor-block" />
        </span>

        <nav className="flex h-full">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                cn(
                  "h-full flex items-center px-4 font-mono text-data-base transition-colors",
                  isActive
                    ? "text-mcp border-b-2 border-mcp font-semibold"
                    : "text-foreground-muted hover:text-foreground hover:bg-muted"
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="flex items-center gap-3">
        {user && (
          <>
            <span className="font-mono text-label-caps uppercase tracking-wider text-foreground-muted px-2 py-0.5 border border-border rounded">
              {user.role}
            </span>
            <span className="font-mono text-data-sm text-foreground-subtle hidden sm:block">
              {user.email}
            </span>
            <div className="w-px h-4 bg-border mx-1" />
          </>
        )}
        <button
          onClick={logout}
          className="flex items-center gap-1.5 font-mono text-data-sm text-foreground-muted hover:text-foreground transition-colors"
        >
          Sign Out
          <LogOut size={13} />
        </button>
      </div>
    </header>
  );
}
