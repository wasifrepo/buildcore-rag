import { formatScore, scoreColour } from "../utils/formatters";

/**
 * Evaluation dashboard showing system vs baseline comparison and per-item table.
 *
 * Props:
 *   - summary: evaluation_complete SSE payload (or EvaluationReport from disk)
 *     { total_questions, system_scores, baseline_scores, delta, pass_rate }
 *   - cases: array of case_complete payloads
 */
export default function EvalDashboard({ summary, cases }) {
  if (!summary && cases.length === 0) return null;

  return (
    <div className="space-y-6">
      {summary && <SummaryCards summary={summary} />}
      {cases.length > 0 && <CasesTable cases={cases} />}
    </div>
  );
}

function SummaryCards({ summary }) {
  const { system_scores, baseline_scores, delta, pass_rate, total_questions } = summary;

  return (
    <div className="space-y-4">
      {/* Top-line stats */}
      <div className="grid grid-cols-3 gap-4">
        <StatCard
          label="Questions Evaluated"
          value={total_questions ?? "—"}
          sub=""
        />
        <StatCard
          label="Pass Rate"
          value={pass_rate != null ? formatScore(pass_rate) : "—"}
          sub="system overall ≥ 70%"
          valueClass={pass_rate >= 0.7 ? "text-green-400" : "text-red-400"}
        />
        <StatCard
          label="System vs Baseline"
          value={
            delta != null
              ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}pp`
              : "—"
          }
          sub="overall delta"
          valueClass={delta >= 0 ? "text-green-400" : "text-red-400"}
        />
      </div>

      {/* System vs Baseline detail */}
      <div className="grid grid-cols-2 gap-4">
        <ScoreSetCard title="System (Full Pipeline)" scores={system_scores} highlight />
        <ScoreSetCard title="Naive Baseline" scores={baseline_scores} />
      </div>
    </div>
  );
}

function StatCard({ label, value, sub, valueClass = "text-gray-100" }) {
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-900 px-5 py-4">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${valueClass}`}>{value}</p>
      {sub && <p className="text-xs text-gray-600 mt-0.5">{sub}</p>}
    </div>
  );
}

function ScoreSetCard({ title, scores, highlight }) {
  if (!scores) return null;
  const border = highlight ? "border-orange-800" : "border-gray-700";
  const metrics = [
    { key: "avg_faithfulness", label: "Faithfulness" },
    { key: "avg_citation_presence", label: "Citation Presence" },
    { key: "avg_refusal_accuracy", label: "Refusal Accuracy" },
    { key: "overall", label: "Overall" },
  ];

  return (
    <div className={`rounded-xl border ${border} bg-gray-900 px-5 py-4`}>
      <p className="text-sm font-semibold text-gray-300 mb-3">{title}</p>
      <div className="space-y-2">
        {metrics.map(({ key, label }) => (
          <div key={key} className="flex items-center justify-between">
            <span className="text-xs text-gray-500">{label}</span>
            <div className="flex items-center gap-2">
              <div className="w-24 bg-gray-800 rounded-full h-1.5">
                <div
                  className="h-1.5 rounded-full bg-orange-500"
                  style={{ width: `${Math.round((scores[key] ?? 0) * 100)}%` }}
                />
              </div>
              <span className={`text-xs font-medium w-10 text-right ${scoreColour(scores[key])}`}>
                {formatScore(scores[key])}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CasesTable({ cases }) {
  return (
    <div>
      <p className="text-sm font-semibold text-gray-400 mb-2">
        Per-item results ({cases.length})
      </p>
      <div className="rounded-xl border border-gray-700 overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-700 bg-gray-800/60">
              <th className="text-left px-4 py-2.5 text-gray-500 font-medium">ID</th>
              <th className="text-left px-4 py-2.5 text-gray-500 font-medium">Question</th>
              <th className="text-left px-4 py-2.5 text-gray-500 font-medium">Difficulty</th>
              <th className="text-right px-4 py-2.5 text-gray-500 font-medium">System</th>
              <th className="text-right px-4 py-2.5 text-gray-500 font-medium">Baseline</th>
              <th className="text-center px-4 py-2.5 text-gray-500 font-medium">Pass</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((c) => (
              <tr
                key={c.id}
                className="border-b border-gray-800 hover:bg-gray-800/30 transition-colors"
              >
                <td className="px-4 py-2.5 font-mono text-gray-500">{c.id}</td>
                <td className="px-4 py-2.5 text-gray-300 max-w-xs">
                  <span className="line-clamp-1">{c.question}</span>
                </td>
                <td className="px-4 py-2.5">
                  <DifficultyBadge difficulty={c.difficulty} />
                </td>
                <td className={`px-4 py-2.5 text-right font-medium ${scoreColour(c.system_score)}`}>
                  {formatScore(c.system_score)}
                </td>
                <td className={`px-4 py-2.5 text-right font-medium ${scoreColour(c.baseline_score)}`}>
                  {formatScore(c.baseline_score)}
                </td>
                <td className="px-4 py-2.5 text-center">
                  {c.passed ? (
                    <span className="text-green-400">✓</span>
                  ) : (
                    <span className="text-red-400">✗</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const DIFFICULTY_STYLES = {
  factual: "bg-blue-900/50 text-blue-300",
  procedural: "bg-purple-900/50 text-purple-300",
  multi_hop: "bg-yellow-900/50 text-yellow-300",
  out_of_scope: "bg-red-900/50 text-red-300",
};

function DifficultyBadge({ difficulty }) {
  const cls = DIFFICULTY_STYLES[difficulty] ?? "bg-gray-800 text-gray-400";
  return (
    <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${cls}`}>
      {difficulty}
    </span>
  );
}
