import { useState, useCallback } from "react";
import { useSSE } from "./useSSE";
import { runEvaluation, getLatestEval } from "../utils/api";

/**
 * Hook that manages running the evaluation suite and loading the latest report.
 *
 * Returns:
 *   - cases: array of case_complete payloads received during the current run
 *   - summary: evaluation_complete payload from the current run (or null)
 *   - savedReport: the latest persisted EvaluationReport (or null)
 *   - isRunning: true while the evaluation stream is open
 *   - error: error string or null
 *   - run(): starts a new evaluation run
 *   - loadLatest(): fetches the most recently saved report from disk
 */
export function useEvaluation() {
  const { steps, isStreaming, error, start, reset } = useSSE();
  const [savedReport, setSavedReport] = useState(null);
  const [loadError, setLoadError] = useState(null);

  const cases = steps
    .filter((s) => s.step === "case_complete")
    .map((s) => s.payload);

  const summaryStep = steps.find((s) => s.step === "evaluation_complete");
  const summary = summaryStep?.payload ?? null;

  const run = useCallback(async () => {
    reset();
    setSavedReport(null);
    await start(
      () => fetch("/evaluate/run", { method: "POST" }),
    );
  }, [reset, start]);

  const loadLatest = useCallback(async () => {
    setLoadError(null);
    try {
      const data = await getLatestEval();
      setSavedReport(data);
    } catch (err) {
      setLoadError(err.message ?? "Failed to load report");
    }
  }, []);

  return {
    cases,
    summary,
    savedReport,
    isRunning: isStreaming,
    error: error ?? loadError,
    run,
    loadLatest,
  };
}
