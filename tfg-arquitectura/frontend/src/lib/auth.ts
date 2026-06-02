import { useEffect, useState } from "react";

import type { User } from "./types";

function readUser(): User | null {
  const raw = localStorage.getItem("tfg_user");
  if (!raw) return null;
  try {
    return JSON.parse(raw) as User;
  } catch {
    return null;
  }
}

export function useAuth() {
  const [token, setToken] = useState<string | null>(() =>
    localStorage.getItem("tfg_token")
  );
  const [user, setUser] = useState<User | null>(() => readUser());

  useEffect(() => {
    const onStorage = () => {
      setToken(localStorage.getItem("tfg_token"));
      setUser(readUser());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const logout = () => {
    localStorage.removeItem("tfg_token");
    localStorage.removeItem("tfg_user");
    setToken(null);
    setUser(null);
    window.location.href = "/login";
  };

  return { token, user, logout };
}
