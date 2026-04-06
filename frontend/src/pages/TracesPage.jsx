import { useTraces } from "../hooks/useTraces";
import TraceExplorer from "../components/TraceExplorer";

export default function TracesPage() {
  const {
    traces,
    selectedTrace,
    loading,
    detailLoading,
    error,
    refresh,
    select,
    clearSelection,
  } = useTraces();

  return (
    <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-100">Trace History</h1>
          <p className="text-sm text-gray-500 mt-1">
            All stored pipeline runs, newest first.
          </p>
        </div>
        <button
          onClick={refresh}
          disabled={loading}
          className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 border border-gray-700 hover:border-gray-600 rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {detailLoading && (
        <p className="text-sm text-gray-500 animate-pulse">Loading trace…</p>
      )}

      {!loading && !detailLoading && (
        <TraceExplorer
          traces={traces}
          selectedTrace={selectedTrace}
          detailLoading={detailLoading}
          onSelect={select}
          onClear={clearSelection}
        />
      )}

      {loading && (
        <p className="text-sm text-gray-500 animate-pulse">Loading traces…</p>
      )}
    </div>
  );
}
