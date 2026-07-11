import { useState, useEffect, useCallback } from "react";
import { useSSE } from "../hooks/useSSE";
import { getLatestEval, BASE_URL } from "../utils/api";
import { formatScore, truncate } from "../utils/formatters";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ScoreBar({ score, accent }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ width: 60, height: 4, background: "var(--surface-elevated)", borderRadius: 4, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${Math.round((score ?? 0) * 100)}%`,
          background: accent ? "var(--accent)" : "var(--text-muted)",
          borderRadius: 4,
          transition: "width 0.4s ease",
        }} />
      </div>
      <span style={{ fontSize: 12, color: accent ? "var(--accent-light)" : "var(--text-secondary)", fontVariantNumeric: "tabular-nums" }}>
        {formatScore(score)}
      </span>
    </div>
  );
}

function MetricRow({ label, score, accent }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 0" }}>
      <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>{label}</span>
      <ScoreBar score={score} accent={accent} />
    </div>
  );
}

function ScoreCard({ title, scores, accent }) {
  const overallPct = Math.round((scores?.overall ?? 0) * 100);
  return (
    <div style={{
      flex: 1,
      background: "var(--surface)",
      border: `1px solid ${accent ? "var(--accent)" : "var(--border)"}`,
      borderRadius: 12,
      padding: 24,
    }}>
      <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", marginBottom: 12 }}>
        {title}
      </div>
      <div style={{
        fontSize: 48,
        fontWeight: 700,
        color: accent ? "var(--accent-light)" : "var(--text-primary)",
        lineHeight: 1,
        marginBottom: 8,
      }}>
        {overallPct}%
      </div>
      <div style={{ height: 4, background: "var(--surface-elevated)", borderRadius: 4, marginBottom: 16, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${overallPct}%`,
          background: accent ? "var(--accent)" : "var(--text-muted)",
          borderRadius: 4,
          transition: "width 0.6s ease",
        }} />
      </div>
      <div style={{ borderTop: "1px solid var(--border)", paddingTop: 12 }}>
        <MetricRow label="Faithfulness" score={scores?.avg_faithfulness} accent={accent} />
        <MetricRow label="Citations" score={scores?.avg_citation_presence} accent={accent} />
        <MetricRow label="Refusals" score={scores?.avg_refusal_accuracy} accent={accent} />
      </div>
    </div>
  );
}

const DIFFICULTY_STYLES = {
  factual: { background: "rgba(99,102,241,0.15)", color: "var(--accent-light)" },
  procedural: { background: "rgba(168,85,247,0.15)", color: "#c084fc" },
  multi_hop: { background: "rgba(245,158,11,0.15)", color: "var(--warning)" },
  out_of_scope: { background: "rgba(71,85,105,0.15)", color: "var(--text-muted)" },
};

function DifficultyPill({ difficulty }) {
  const style = DIFFICULTY_STYLES[difficulty] ?? DIFFICULTY_STYLES.out_of_scope;
  return (
    <span style={{
      ...style,
      borderRadius: 999,
      padding: "2px 8px",
      fontSize: 11,
      fontWeight: 600,
      whiteSpace: "nowrap",
    }}>
      {difficulty}
    </span>
  );
}

function PassIcon({ passed }) {
  if (passed) {
    return (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M2.5 7L5.5 10L11.5 4" stroke="var(--success)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M3 3L11 11M11 3L3 11" stroke="var(--danger)" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function ResultsTable({ cases }) {
  return (
    <div style={{ marginTop: 32 }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            {["Question", "Difficulty", "System", "Baseline", "Pass"].map((h) => (
              <th key={h} style={{
                textAlign: "left",
                padding: "0 0 10px 0",
                fontSize: 11,
                color: "var(--text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                fontWeight: 600,
                borderBottom: "1px solid var(--border)",
              }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cases.map((c, i) => (
            <ResultRow key={c.id ?? i} item={c} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultRow({ item }) {
  const [hovered, setHovered] = useState(false);
  return (
    <tr
      style={{
        borderBottom: "1px solid var(--border)",
        background: hovered ? "var(--surface-elevated)" : "transparent",
        transition: "background 0.15s",
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <td style={{ padding: "12px 0", fontSize: 13, color: "var(--text-primary)", maxWidth: 320 }}
          title={item.question}>
        {truncate(item.question, 60)}
      </td>
      <td style={{ padding: "12px 8px" }}>
        <DifficultyPill difficulty={item.difficulty} />
      </td>
      <td style={{ padding: "12px 8px" }}>
        <ScoreBar score={item.system_score ?? item.system_overall} accent />
      </td>
      <td style={{ padding: "12px 8px" }}>
        <ScoreBar score={item.baseline_score ?? item.baseline_overall} accent={false} />
      </td>
      <td style={{ padding: "12px 0", textAlign: "center" }}>
        <PassIcon passed={item.passed} />
      </td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function EvalPage() {
  const { steps, isStreaming, error: sseError, start, reset } = useSSE();
  const [savedReport, setSavedReport] = useState(null);
  const [loadError, setLoadError] = useState(null);

  // Extract streaming results
  const streamedCases = steps
    .filter((s) => s.step === "case_complete")
    .map((s) => s.payload);
  const summaryStep = steps.find((s) => s.step === "evaluation_complete");
  const streamedSummary = summaryStep?.payload ?? null;

  // Resolve display data: prefer live stream, fall back to saved
  const activeSummary = streamedSummary ?? savedReport;
  const activeCases = streamedCases.length > 0
    ? streamedCases
    : (savedReport?.per_item_results ?? []).map((r) => ({
        id: r.id,
        question: r.question,
        difficulty: r.difficulty,
        system_score: r.system_overall,
        baseline_score: r.baseline_overall,
        passed: r.passed,
      }));

  const hasResults = activeSummary || activeCases.length > 0;

  const loadLatest = useCallback(async () => {
    setLoadError(null);
    try {
      const data = await getLatestEval();
      if (data) setSavedReport(data);
    } catch (err) {
      setLoadError(err.message ?? "Failed to load report");
    }
  }, []);

  useEffect(() => { loadLatest(); }, [loadLatest]);

  async function handleRun() {
    reset();
    setSavedReport(null);
    await start(() => fetch(`${BASE_URL}/evaluate/run`, { method: "POST" }));
  }

  const delta = activeSummary
    ? (activeSummary.delta ?? activeSummary.system_scores?.overall - activeSummary.baseline_scores?.overall)
    : null;

  const error = sseError ?? loadError;

  return (
    <div>
      {/* Header */}
      <h1 style={{ fontSize: 24, fontWeight: 600, color: "var(--text-primary)" }}>
        Evaluation Dashboard
      </h1>
      <p style={{ fontSize: 14, color: "var(--text-secondary)", marginTop: 6, marginBottom: 32 }}>
        System performance vs naive RAG baseline across 50 test questions.
      </p>

      {/* Run button / progress */}
      {!hasResults && !isStreaming && (
        <button
          onClick={handleRun}
          style={{
            width: "100%",
            background: "var(--accent)",
            color: "white",
            border: "none",
            borderRadius: 8,
            padding: "14px 24px",
            fontSize: 14,
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Run Evaluation Suite
        </button>
      )}

      {isStreaming && (
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>
              Evaluating question {streamedCases.length} of 50
            </span>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
              {Math.round((streamedCases.length / 50) * 100)}%
            </span>
          </div>
          <div style={{ height: 3, background: "var(--border)", borderRadius: 4, overflow: "hidden" }}>
            <div style={{
              height: "100%",
              width: `${(streamedCases.length / 50) * 100}%`,
              background: "var(--accent)",
              borderRadius: 4,
              transition: "width 0.3s ease",
            }} />
          </div>
        </div>
      )}

      {error && (
        <div style={{
          background: "rgba(239,68,68,0.08)",
          border: "1px solid rgba(239,68,68,0.2)",
          borderRadius: 8,
          padding: "10px 14px",
          fontSize: 13,
          color: "var(--danger)",
          marginTop: 16,
        }}>
          {error}
        </div>
      )}

      {/* Comparison cards */}
      {activeSummary && (
        <>
          <div style={{ display: "flex", gap: 16, marginTop: hasResults && !isStreaming ? 0 : 24 }}>
            <ScoreCard
              title="BuildCore Intelligence"
              scores={activeSummary.system_scores}
              accent
            />
            <ScoreCard
              title="Naive Baseline"
              scores={activeSummary.baseline_scores}
              accent={false}
            />
          </div>

          {delta != null && (
            <div style={{ textAlign: "center", marginTop: 20 }}>
              <div style={{ fontSize: 20, fontWeight: 700, color: delta >= 0 ? "var(--success)" : "var(--danger)" }}>
                {delta >= 0 ? "+" : ""}{(delta * 100).toFixed(1)}pp improvement
              </div>
              <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 4 }}>
                overall score delta vs naive baseline
              </div>
            </div>
          )}

          {hasResults && !isStreaming && (
            <div style={{ marginTop: 24, textAlign: "right" }}>
              <button
                onClick={handleRun}
                style={{
                  background: "none",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  padding: "8px 16px",
                  fontSize: 13,
                  color: "var(--text-secondary)",
                  cursor: "pointer",
                }}
              >
                Re-run evaluation
              </button>
            </div>
          )}
        </>
      )}

      {/* Results table */}
      {activeCases.length > 0 && <ResultsTable cases={activeCases} />}
    </div>
  );
}
