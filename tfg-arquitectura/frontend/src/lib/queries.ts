import { useMemo } from "react";
import { useQuery, useQueries, useMutation, useQueryClient } from "@tanstack/react-query";

import { api, listParams } from "./api";
import type {
  AuthResponse,
  ComparisonRow,
  Dataset,
  Experiment,
  ExperimentCreate,
  ExperimentDetail,
  McpContext,
  Metric,
  ModelCatalog,
  Narration,
  Observation,
  Prediction,
  Run,
  Series,
  WhatifSetup,
} from "./types";

// ── auth ─────────────────────────────────────────────────────────────────────

export function useLogin() {
  return useMutation({
    mutationFn: async (creds: { email: string; password: string }) =>
      api.post<AuthResponse>("/auth/login", creds),
    onSuccess: (data) => {
      localStorage.setItem("tfg_token", data.access_token);
      localStorage.setItem("tfg_user", JSON.stringify(data.user));
    },
  });
}

export function useSignup() {
  return useMutation({
    mutationFn: async (creds: { email: string; password: string }) =>
      api.post<AuthResponse>("/auth/signup", creds),
    onSuccess: (data) => {
      localStorage.setItem("tfg_token", data.access_token);
      localStorage.setItem("tfg_user", JSON.stringify(data.user));
    },
  });
}

// ── datasets / catalog ───────────────────────────────────────────────────────

export function useDatasets() {
  return useQuery({
    queryKey: ["datasets"],
    queryFn: () => api.get<Dataset[]>("/datasets"),
  });
}

export function useSeries(datasetId: number | null) {
  return useQuery({
    queryKey: ["datasets", datasetId, "series"],
    queryFn: () => api.get<Series[]>(`/datasets/${datasetId}/series`),
    enabled: datasetId !== null,
  });
}

export function useObservations(seriesId: number | undefined, limit = 1000) {
  return useQuery({
    queryKey: ["series", seriesId, "observations", limit],
    queryFn: () => api.get<Observation[]>(`/series/${seriesId}/observations?limit=${limit}`),
    enabled: seriesId !== undefined,
    staleTime: 60_000,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: () => api.get<ModelCatalog[]>("/models"),
  });
}

export interface SeriesInfo {
  id: number;
  name: string;
  datasetId: number;
  datasetName: string;
}

/** Resolve every series across all datasets into a single id → info map. */
export function useAllSeries(): { map: Map<number, SeriesInfo>; isLoading: boolean } {
  const datasets = useDatasets();
  const dsList = datasets.data ?? [];

  const results = useQueries({
    queries: dsList.map((d) => ({
      queryKey: ["datasets", d.id, "series"],
      queryFn: () => api.get<Series[]>(`/datasets/${d.id}/series`),
      staleTime: 5 * 60_000,
    })),
  });

  const map = useMemo(() => {
    const m = new Map<number, SeriesInfo>();
    dsList.forEach((d, i) => {
      (results[i]?.data ?? []).forEach((s) =>
        m.set(s.id, { id: s.id, name: s.name, datasetId: d.id, datasetName: d.name })
      );
    });
    return m;
  }, [dsList, results]);

  const isLoading = datasets.isLoading || results.some((r) => r.isLoading);
  return { map, isLoading };
}

// ── experiments ──────────────────────────────────────────────────────────────

export function useExperiments() {
  return useQuery({
    queryKey: ["experiments"],
    queryFn: () => api.get<Experiment[]>("/experiments"),
  });
}

export function useExperiment(id: number | undefined) {
  return useQuery({
    queryKey: ["experiments", id],
    queryFn: () => api.get<ExperimentDetail>(`/experiments/${id}`),
    enabled: id !== undefined,
  });
}

export function useCreateExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: ExperimentCreate) =>
      api.post<Experiment>("/experiments", payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["experiments"] }),
  });
}

export function useDeleteExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.delete(`/experiments/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["experiments"] }),
  });
}

// ── runs ─────────────────────────────────────────────────────────────────────

export function useExperimentRuns(experimentId: number | undefined) {
  return useQuery({
    queryKey: ["experiments", experimentId, "runs"],
    queryFn: () => api.get<Run[]>(`/experiments/${experimentId}/runs`),
    enabled: experimentId !== undefined,
    refetchInterval: (q) => {
      const rows = q.state.data as Run[] | undefined;
      return rows?.some((r) => r.status === "pending" || r.status === "running")
        ? 2000
        : false;
    },
  });
}

export function useTriggerRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (experimentId: number) =>
      api.post<Run>(`/experiments/${experimentId}/runs`, {}),
    onSuccess: (_data, experimentId) => {
      qc.invalidateQueries({ queryKey: ["experiments", experimentId, "runs"] });
      qc.invalidateQueries({ queryKey: ["experiments", experimentId] });
    },
  });
}

export function useRun(id: number | undefined) {
  return useQuery({
    queryKey: ["runs", id],
    queryFn: () => api.get<Run>(`/runs/${id}`),
    enabled: id !== undefined,
    refetchInterval: (q) => {
      const r = q.state.data as Run | undefined;
      return r && (r.status === "pending" || r.status === "running") ? 2000 : false;
    },
  });
}

export function usePredictions(runId: number | undefined) {
  return useQuery({
    queryKey: ["runs", runId, "predictions"],
    queryFn: () => api.get<Prediction[]>(`/runs/${runId}/predictions?limit=500`),
    enabled: runId !== undefined,
  });
}

export function useRunMetrics(runId: number | undefined) {
  return useQuery({
    queryKey: ["runs", runId, "metrics"],
    queryFn: () => api.get<Metric[]>(`/runs/${runId}/metrics`),
    enabled: runId !== undefined,
  });
}

export function useMcpContext(runId: number | undefined) {
  return useQuery({
    queryKey: ["runs", runId, "mcp-context"],
    queryFn: () => api.get<McpContext>(`/runs/${runId}/mcp-context`),
    enabled: runId !== undefined,
    retry: false,
  });
}

export function useNarration(runId: number | undefined) {
  return useMutation({
    mutationFn: () =>
      api.post<Narration>(`/runs/${runId}/narration`, {}),
    onError: () => {},
  });
}

// ── drift detection ──────────────────────────────────────────────────────────

export interface DriftResult {
  experiment_id: number;
  run_id: number | null;
  drifted: boolean;
  p_value: number | null;
  ks_statistic: number | null;
  n_early: number;
  n_recent: number;
  message: string;
}

export function useDrift(experimentId: number | undefined) {
  return useQuery({
    queryKey: ["drift", experimentId],
    queryFn: () => api.get<DriftResult>(`/drift?experiment_id=${experimentId}`),
    enabled: experimentId !== undefined,
    staleTime: 60_000,
    retry: false,
  });
}

// ── what-if simulator ────────────────────────────────────────────────────────

export function useWhatifSetup(seriesId: number | undefined, horizon: number) {
  return useQuery({
    queryKey: ["whatif", seriesId, horizon],
    queryFn: () =>
      api.get<WhatifSetup>(`/whatif/setup?series_id=${seriesId}&horizon=${horizon}`),
    enabled: seriesId !== undefined,
    staleTime: 60_000,
    retry: false,
  });
}

// ── live news pulse (GDELT + FinBERT via MCP) ────────────────────────────────

export interface NewsArticle {
  title: string;
  date: string;
  source: string;
  url: string;
  language?: string;
}

export interface NewsToday {
  articles: NewsArticle[];
  total: number;
  latest_month: string | null;
  last_refresh: string | null;
}

export interface NewsSentiment {
  available: boolean;
  year_month?: string;
  country?: string;
  n_articles?: number;
  sentiment_mean?: number | null;
  sentiment_std?: number | null;
  hawkish_score?: number | null;
  message?: string;
}

export function useNewsToday(limit = 12) {
  return useQuery({
    queryKey: ["news", "today", limit],
    queryFn: () => api.get<NewsToday>(`/news/today?limit=${limit}`),
    staleTime: 60_000,
    retry: false,
  });
}

export function useNewsSentiment(yearMonth: string | null | undefined) {
  return useQuery({
    queryKey: ["news", "sentiment", yearMonth],
    queryFn: () => api.get<NewsSentiment>(`/news/sentiment?year_month=${yearMonth}`),
    enabled: !!yearMonth,
    staleTime: 5 * 60_000,
    retry: false,
  });
}

export function useRefreshNews() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (timespan: string) =>
      api.post<{ ingested: number; total: number }>(
        `/news/refresh?timespan=${timespan}`,
        {}
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["news"] }),
  });
}

// ── Diebold-Mariano significance ─────────────────────────────────────────────

export interface DmRunMeta {
  run_id: number;
  experiment_id: number;
  experiment_name: string;
  model_slug: string;
  model_name: string;
  horizon: number;
  use_mcp: boolean;
  mae: number | null;
  n_points: number;
}

export interface DmPair {
  a_run_id: number;
  b_run_id: number;
  comparable: boolean;
  reason?: string;
  dm_stat?: number;
  p_value?: number;
  better?: "model1" | "model2" | "tie";
  significant?: boolean;
  n?: number;
}

export interface DmMatrixResult {
  power: number;
  alpha: number;
  runs: DmRunMeta[];
  pairs: DmPair[];
}

export function useDmMatrix(runIds: number[]) {
  return useQuery({
    queryKey: ["dm-matrix", runIds],
    queryFn: () =>
      api.get<DmMatrixResult>(`/metrics/dm-matrix?${listParams("run_ids", runIds)}`),
    enabled: runIds.length >= 2,
    staleTime: 60_000,
    retry: false,
  });
}

// ── metrics comparison ───────────────────────────────────────────────────────

export function useComparison(experimentIds: number[]) {
  return useQuery({
    queryKey: ["compare", experimentIds],
    queryFn: () =>
      api.get<ComparisonRow[]>(
        `/metrics/compare?${listParams("experiment_ids", experimentIds)}`
      ),
    enabled: experimentIds.length > 0,
  });
}
