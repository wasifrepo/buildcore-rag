import { useState } from "react";
import { formatLatency, formatTimestamp, formatScore, truncate, scoreColour } from "../utils/formatters";

/**
 * Trace explorer: searchable list + expandable full-trace detail.
 *
 * Props:
 *   - traces: array of TraceSummary
 *   - selectedTrace: full trace object or null
 *   - detailLoading: bool
 *   - onSelect(traceId): called when a row is clicked
 *   - onClear(): called to deselect
 */
export default function TraceExplorer({
  traces,
  selectedTrace,
  detailLoading,
  onSelect,
  onClear,
}) {
  const [search, setSearch] = useState("");

  const filtered = traces.filter((t) =>
    t.question.toLowerCase().includes(search.toLowerCase())
  );

  if (selectedTrace) {
    return (
      <TraceDetail
        data={selectedTrace}
        onBack={onClear}
      />
    );
  }

  return (
    <div className="space-y-4">
      {/* Search */}
      <input
        type="text"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Filter by question…"
        className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-orange-500"
      />

      {/* Table */}
      {filtered.length === 0 ? (
        <p className="text-gray-500 text-sm">No traces found.</p>
      ) : (
        <div className="rounded-xl border border-gray-700 overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-700 bg-gray-800/60">
                <th className="text-left px-4 py-2.5 text-gray-500 font-medium">Question</th>
                <th className="text-right px-4 py-2.5 text-gray-500 font-medium">Latency</th>
                <th className="text-center px-4 py-2.5 text-gray-500 font-medium">Passes</th>
                <th className="text-right px-4 py-2.5 text-gray-500 font-medium">Confidence</th>
                <th className="text-left px-4 py-2.5 text-gray-500 font-medium">Time</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((t) => (
                <tr
                  key={t.trace_id}
                  onClick={() => onSelect(t.trace_id)}
                  className="border-b border-gray-800 hover:bg-gray-800/40 cursor-pointer transition-colors"
                >
                  <td className="px-4 py-3 text-gray-200">
                    <span className="line-clamp-1">{t.question}</span>
                    {t.answer_refused && (
                      <span className="ml-2 text-xs text-red-400">[refused]</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right text-gray-400 font-mono">
                    {formatLatency(t.total_latency_ms)}
                  </td>
                  <td className="px-4 py-3 text-center text-gray-400">
                    {t.retrieval_passes}
                    {t.second_pass_triggered && (
                      <span className="ml-1 text-yellow-500">↺</span>
                    )}
                  </td>
                  <td className={`px-4 py-3 text-right font-medium ${scoreColour(t.answer_confidence)}`}>
                    {formatScore(t.answer_confidence)}
                  </td>
                  <td className="px-4 py-3 text-gray-500">
                    {formatTimestamp(t.timestamp)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function TraceDetail({ data, onBack }) {
  const [expanded, setExpanded] = useState(false);
  const trace = data.trace ?? {};

  return (
    <div className="space-y-4">
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-200 transition-colors"
      >
        ← Back to list
      </button>

      {/* Summary header */}
      <div className="rounded-xl border border-gray-700 bg-gray-900 px-5 py-4">
        <p className="text-sm font-semibold text-gray-200 mb-3">{trace.question}</p>
        <div className="grid grid-cols-4 gap-4 text-xs">
          <div>
            <p className="text-gray-500">Latency</p>
            <p className="text-gray-200 font-medium">
              {formatLatency(trace.total_latency_ms)}
            </p>
          </div>
          <div>
            <p className="text-gray-500">Retrieval Passes</p>
            <p className="text-gray-200 font-medium">{trace.retrieval_passes ?? 1}</p>
          </div>
          <div>
            <p className="text-gray-500">Confidence</p>
            <p className={`font-medium ${scoreColour(trace.final_answer?.confidence)}`}>
              {formatScore(trace.final_answer?.confidence)}
            </p>
          </div>
          <div>
            <p className="text-gray-500">Refused</p>
            <p className={trace.final_answer?.refused ? "text-red-400" : "text-green-400"}>
              {trace.final_answer?.refused ? "Yes" : "No"}
            </p>
          </div>
        </div>
      </div>

      {/* Answer */}
      {trace.final_answer?.answer && (
        <div className="rounded-xl border border-gray-700 bg-gray-900 px-5 py-4">
          <p className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-2">
            Answer
          </p>
          <p className="text-sm text-gray-200 whitespace-pre-wrap leading-relaxed">
            {trace.final_answer.answer}
          </p>
        </div>
      )}

      {/* Raw JSON toggle */}
      <div className="rounded-xl border border-gray-700 overflow-hidden">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="w-full flex items-center justify-between px-5 py-3 bg-gray-800/60 text-sm text-gray-400 hover:text-gray-200 transition-colors"
        >
          <span>Raw JSON trace</span>
          <span>{expanded ? "▲" : "▼"}</span>
        </button>
        {expanded && (
          <pre className="px-5 py-4 text-xs text-gray-400 overflow-x-auto bg-gray-950 max-h-96">
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
