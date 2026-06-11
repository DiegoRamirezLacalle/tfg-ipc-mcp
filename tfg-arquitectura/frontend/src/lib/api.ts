// Thin fetch wrapper. The dev server proxies /api to the FastAPI backend (see vite.config.ts).

const BASE = "/api/v1";

function authHeaders(): HeadersInit {
  const token = localStorage.getItem("tfg_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(
  path: string,
  init: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(init.headers ?? {}),
    },
  });

  if (res.status === 401) {
    localStorage.removeItem("tfg_token");
    window.location.href = "/login";
    throw new Error("Unauthorized");
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  get:    <T>(path: string)              => request<T>(path),
  post:   <T>(path: string, body: unknown) => request<T>(path, { method: "POST",   body: JSON.stringify(body) }),
  patch:  <T>(path: string, body: unknown) => request<T>(path, { method: "PATCH",  body: JSON.stringify(body) }),
  delete: <T>(path: string)              => request<T>(path, { method: "DELETE" }),
};

// Helper to build query strings for endpoints that take list params (e.g. /metrics/compare).
export function listParams(key: string, values: number[]): string {
  const usp = new URLSearchParams();
  values.forEach((v) => usp.append(key, String(v)));
  return usp.toString();
}
