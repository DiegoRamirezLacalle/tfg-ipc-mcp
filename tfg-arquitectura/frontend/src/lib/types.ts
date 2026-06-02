// TypeScript mirrors of the FastAPI Pydantic schemas (app/schemas/*.py).
// Keep in sync manually — they are small and stable.

export type UserRole = "admin" | "researcher" | "viewer";

export interface User {
  id: number;
  email: string;
  role: UserRole;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Dataset {
  id: number;
  slug: string;
  name: string;
  frequency: string;
}

export interface Series {
  id: number;
  dataset_id: number;
  slug: string;
  name: string;
  unit: string | null;
}

export interface ModelCatalog {
  id: number;
  slug: string;
  name: string;
  model_type: string;
  description: string | null;
  supports_mcp: boolean;
  is_active: boolean;
}

export type ExperimentStatus = "created" | "running" | "done" | "failed";
export type RunStatus = "pending" | "running" | "done" | "failed";

export interface Experiment {
  id: number;
  user_id: number;
  name: string;
  series_id: number;
  model_id: number;
  horizon: number;
  use_mcp: boolean;
  config: Record<string, unknown> | null;
  status: ExperimentStatus;
  created_at: string;
  updated_at: string;
}

export interface Run {
  id: number;
  experiment_id: number;
  status: RunStatus;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  created_at: string;
}

export interface ExperimentDetail extends Experiment {
  runs: Run[];
}

export interface Prediction {
  id: number;
  timestamp: string;
  value: number;
  lower_ci: number | null;
  upper_ci: number | null;
}

export interface Observation {
  id: number;
  timestamp: string;
  value: number;
}

export interface Metric {
  id: number;
  name: string;
  value: number;
}

export interface MetricValues {
  mae: number | null;
  rmse: number | null;
  mape: number | null;
}

export interface ComparisonRow {
  experiment_id: number;
  experiment_name: string;
  model_slug: string;
  model_name: string;
  horizon: number;
  use_mcp: boolean;
  run_id: number | null;
  run_finished_at: string | null;
  metrics: MetricValues | null;
}

export interface McpContext {
  run_id: number;
  fetched_at: string;
  signals: Array<{
    year_month: string;
    [key: string]: unknown;
  }>;
}

export interface Narration {
  narrative: string;
  model: string;
}

export interface ExperimentCreate {
  name: string;
  series_id: number;
  model_id: number;
  horizon?: number;
  use_mcp?: boolean;
  config?: Record<string, unknown> | null;
}

export interface WhatifPoint {
  timestamp: string;
  value: number;
}

export interface WhatifSignal {
  key: string;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  baseline_value: number;
  effect_per_step: number[];
}

export interface WhatifSetup {
  series_id: number;
  series_name: string;
  unit: string | null;
  horizon: number;
  history: WhatifPoint[];
  baseline: WhatifPoint[];
  signals: WhatifSignal[];
  signals_available: boolean;
}
