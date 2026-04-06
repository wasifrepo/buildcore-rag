import { useState, useCallback, useEffect } from "react";
import { listTraces, getTrace } from "../utils/api";

/**
 * Hook for loading and browsing stored pipeline traces.
 *
 * Returns:
 *   - traces: array of TraceSummary objects (newest first)
 *   - selectedTrace: full trace object for the selected trace, or null
 *   - loading: true while fetching the trace list
 *   - detailLoading: true while fetching a single trace detail
 *   - error: error string or null
 *   - refresh(): re-fetches the trace list
 *   - select(traceId): fetches and sets the full trace for the given ID
 *   - clearSelection(): clears the selected trace
 */
export function useTraces() {
  const [traces, setTraces] = useState([]);
  const [selectedTrace, setSelectedTrace] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listTraces();
      setTraces(data.traces ?? []);
    } catch (err) {
      setError(err.message ?? "Failed to load traces");
    } finally {
      setLoading(false);
    }
  }, []);

  const select = useCallback(async (traceId) => {
    setDetailLoading(true);
    setError(null);
    try {
      const data = await getTrace(traceId);
      setSelectedTrace(data);
    } catch (err) {
      setError(err.message ?? "Failed to load trace");
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedTrace(null);
  }, []);

  // Load on mount
  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    traces,
    selectedTrace,
    loading,
    detailLoading,
    error,
    refresh,
    select,
    clearSelection,
  };
}
