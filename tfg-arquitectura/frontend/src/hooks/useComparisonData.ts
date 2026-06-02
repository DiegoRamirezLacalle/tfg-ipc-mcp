import { useMemo } from "react";
import { useQueries } from "@tanstack/react-query";

import { api, listParams } from "@/lib/api";
import { useExperiments, useModels } from "@/lib/queries";
import type {
  ComparisonRow,
  Observation,
  Prediction,
  Run,
  RunStatus,
} from "@/lib/types";

export interface ComparisonRunData {
  runId: number;
  experimentId: number;
  experimentName: string;
  modelId: number;
  modelSlug: string;
  modelName: string;
  seriesId: number;
  horizon: number;
  useMcp: boolean;
  status: RunStatus | "missing";
  startedAt: string | null;
  finishedAt: string | null;
  durationSec: number | null;
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  skill: number | null;
}

export interface ComparisonData {
  runs: ComparisonRunData[];
  predictionsByRun: Record<number, Prediction[]>;
  actualsBySeries: Record<number, Observation[]>;
  naiveMAEBySeries: Record<number, number | null>;
}

function toDate(iso: string): string {
  return iso.slice(0, 10);
}

function computeSeasonalNaiveMAE(
  actuals: Observation[],
  windowStart: string,
  windowEnd: string,
  seasonality = 12
): number | null {
  if (actuals.length < seasonality + 1) return null;
  const byDate = new Map<string, number>();
  actuals.forEach((o) => byDate.set(toDate(o.timestamp), o.value));
  const sorted = [...actuals].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  let sum = 0;
  let n = 0;
  for (const o of sorted) {
    const d = toDate(o.timestamp);
    if (d < windowStart || d > windowEnd) continue;
    const tIdx = sorted.findIndex((x) => toDate(x.timestamp) === d);
    if (tIdx < seasonality) continue;
    const prev = sorted[tIdx - seasonality].value;
    sum += Math.abs(o.value - prev);
    n += 1;
  }
  return n > 0 ? sum / n : null;
}

interface ComparisonDataResult {
  data: ComparisonData | null;
  isLoading: boolean;
  isFetching: boolean;
}

export function useComparisonData(experimentIds: number[]): ComparisonDataResult {
  const experiments = useExperiments();
  const models = useModels();

  const compareQueries = useQueries({
    queries: [
      {
        queryKey: ["compare", experimentIds],
        queryFn: () =>
          api.get<ComparisonRow[]>(
            `/metrics/compare?${listParams("experiment_ids", experimentIds)}`
          ),
        enabled: experimentIds.length > 0,
        staleTime: 30_000,
      },
    ],
  });
  const compareData = compareQueries[0].data;

  const runIds = useMemo(
    () => (compareData ?? []).map((r) => r.run_id).filter((id): id is number => id !== null),
    [compareData]
  );

  const runQueries = useQueries({
    queries: runIds.map((id) => ({
      queryKey: ["runs", id],
      queryFn: () => api.get<Run>(`/runs/${id}`),
      enabled: id !== undefined,
      staleTime: 30_000,
    })),
  });

  const predictionsQueries = useQueries({
    queries: runIds.map((id) => ({
      queryKey: ["runs", id, "predictions"],
      queryFn: () => api.get<Prediction[]>(`/runs/${id}/predictions?limit=500`),
      enabled: id !== undefined,
      staleTime: 60_000,
    })),
  });

  const expById = useMemo(() => {
    const m = new Map<number, { seriesId: number; horizon: number; useMcp: boolean; name: string; modelId: number }>();
    (experiments.data ?? []).forEach((e) =>
      m.set(e.id, { seriesId: e.series_id, horizon: e.horizon, useMcp: e.use_mcp, name: e.name, modelId: e.model_id })
    );
    return m;
  }, [experiments.data]);

  const seriesIds = useMemo(() => {
    const set = new Set<number>();
    (compareData ?? []).forEach((row) => {
      const exp = expById.get(row.experiment_id);
      if (exp) set.add(exp.seriesId);
    });
    return Array.from(set);
  }, [compareData, expById]);

  const observationsQueries = useQueries({
    queries: seriesIds.map((id) => ({
      queryKey: ["series", id, "observations", 1000],
      queryFn: () => api.get<Observation[]>(`/series/${id}/observations?limit=1000`),
      enabled: id !== undefined,
      staleTime: 5 * 60_000,
    })),
  });

  const isLoading =
    compareQueries.some((q) => q.isLoading) ||
    runQueries.some((q) => q.isLoading) ||
    predictionsQueries.some((q) => q.isLoading) ||
    observationsQueries.some((q) => q.isLoading) ||
    experiments.isLoading ||
    models.isLoading;

  const isFetching =
    compareQueries.some((q) => q.isFetching) ||
    runQueries.some((q) => q.isFetching) ||
    predictionsQueries.some((q) => q.isFetching) ||
    observationsQueries.some((q) => q.isFetching);

  const data = useMemo<ComparisonData | null>(() => {
    if (!compareData || experimentIds.length === 0) return null;
    if (!experiments.data || !models.data) return null;

    const predictionsByRun: Record<number, Prediction[]> = {};
    runIds.forEach((id, i) => {
      const r = predictionsQueries[i]?.data;
      if (r) predictionsByRun[id] = r;
    });

    const runById = new Map<number, Run>();
    runIds.forEach((id, i) => {
      const r = runQueries[i]?.data;
      if (r) runById.set(id, r);
    });

    const actualsBySeries: Record<number, Observation[]> = {};
    seriesIds.forEach((id, i) => {
      const r = observationsQueries[i]?.data;
      if (r) actualsBySeries[id] = r;
    });

    const modelMap = new Map(models.data.map((m) => [m.id, m]));

    const naiveMAEBySeries: Record<number, number | null> = {};

    const runs: ComparisonRunData[] = compareData.map((row) => {
      const exp = expById.get(row.experiment_id);
      const seriesId = exp?.seriesId ?? 0;
      const horizon = exp?.horizon ?? 0;
      const runDetail = row.run_id !== null ? runById.get(row.run_id) : undefined;
      const startedAt = runDetail?.started_at ?? null;
      const finishedAt = runDetail?.finished_at ?? row.run_finished_at ?? null;
      const durationSec =
        startedAt && finishedAt
          ? Math.max(0, (new Date(finishedAt).getTime() - new Date(startedAt).getTime()) / 1000)
          : null;

      // Compute seasonal-naive MAE for this run's prediction window
      const preds = row.run_id !== null ? predictionsByRun[row.run_id] : undefined;
      const actuals = actualsBySeries[seriesId];
      let naiveMae: number | null = naiveMAEBySeries[seriesId] ?? null;
      if (preds && preds.length > 0 && actuals && actuals.length > 12) {
        const windowStart = toDate(preds[0].timestamp);
        const windowEnd = toDate(preds[preds.length - 1].timestamp);
        const cacheKey = `${seriesId}::${windowStart}::${windowEnd}`;
        const cached = (naiveMAEBySeries as Record<string, number | null>)[cacheKey];
        if (cached !== undefined) {
          naiveMae = cached;
        } else {
          naiveMae = computeSeasonalNaiveMAE(actuals, windowStart, windowEnd);
          (naiveMAEBySeries as Record<string, number | null>)[cacheKey] = naiveMae;
          naiveMAEBySeries[seriesId] = naiveMae;
        }
      }

      const mae = row.metrics?.mae ?? null;
      const skill = mae !== null && naiveMae !== null && naiveMae > 0 ? 1 - mae / naiveMae : null;

      return {
        runId: row.run_id ?? -1,
        experimentId: row.experiment_id,
        experimentName: row.experiment_name,
        modelId: exp?.modelId ?? 0,
        modelSlug: row.model_slug,
        modelName: row.model_name,
        seriesId,
        horizon,
        useMcp: exp?.useMcp ?? row.use_mcp,
        status: (runDetail?.status as RunStatus) ?? "missing",
        startedAt,
        finishedAt,
        durationSec,
        mae,
        rmse: row.metrics?.rmse ?? null,
        mape: row.metrics?.mape ?? null,
        skill,
      };
    });

    void modelMap;
    return { runs, predictionsByRun, actualsBySeries, naiveMAEBySeries };
  }, [
    compareData,
    experimentIds.length,
    experiments.data,
    models.data,
    runIds,
    seriesIds,
    expById,
    runQueries,
    predictionsQueries,
    observationsQueries,
  ]);

  return { data, isLoading, isFetching };
}
