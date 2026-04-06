import { useEffect } from "react";
import { useEvaluation } from "../hooks/useEvaluation";
import EvalDashboard from "../components/EvalDashboard";

export default function EvalPage() {
  const { cases, summary, savedReport, isRunning, error, run, loadLatest } =
    useEvaluation();

  // Try to load the latest saved report on mount
  useEffect(() => {
    loadLatest();
  }, [loadLatest]);

  const displaySummary = summary ?? savedReport;
  const displayCases =
    cases.length > 0
      ? cases
      : (savedReport?.per_item_results ?? []).map((r) => ({
          id: r.id,
          question: r.question,
          difficulty: r.difficulty,
          system_score: r.system_overall,
          baseline_score: r.baseline_overall,
          system_faithfulness: r.system_faithfulness,
          baseline_faithfulness: r.baseline_faithfulness,
          system_citation_presence: r.system_citation_presence,
          system_refusal_accuracy: r.system_refusal_accuracy,
          system_refused: r.system_refused,
          passed: r.passed,
        }));

  return (
    <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-100">Evaluation Suite</h1>
          <p className="text-sm text-gray-500 mt-1">
            50-item test suite — full pipeline vs naive baseline.
          </p>
        </div>
        <button
          onClick={run}
          disabled={isRunning}
          className="px-5 py-2.5 bg-orange-500 hover:bg-orange-600 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          {isRunning ? "Running…" : "Run Evaluation"}
        </button>
      </div>

      {/* Progress bar while running */}
      {isRunning && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-500">
            <span>Evaluating…</span>
            <span>{cases.length} / 50</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-1.5">
            <div
              className="h-1.5 rounded-full bg-orange-500 transition-all"
              style={{ width: `${(cases.length / 50) * 100}%` }}
            />
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {displaySummary || displayCases.length > 0 ? (
        <EvalDashboard summary={displaySummary} cases={displayCases} />
      ) : (
        !isRunning && (
          <div className="rounded-xl border border-gray-800 bg-gray-900 px-6 py-10 text-center">
            <p className="text-gray-400 text-sm">
              No evaluation report yet. Click{" "}
              <span className="text-orange-400 font-medium">Run Evaluation</span> to start.
            </p>
          </div>
        )
      )}
    </div>
  );
}
