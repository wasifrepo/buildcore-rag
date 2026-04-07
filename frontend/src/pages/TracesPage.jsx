import { useState, useEffect, useCallback } from "react";
import { listTraces, getTrace } from "../utils/api";
import { formatLatency, formatTimestamp, formatScore, formatConfidence, truncate } from "../utils/formatters";

// ---------------------------------------------------------------------------
// Vertical timeline inside trace detail
// ---------------------------------------------------------------------------

function TimelineStep({ label, detail }) {
  return (
    <div style={{ display: "flex", gap: 16, paddingBottom: 16 }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
        <div style={{
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: "var(--accent)",
          flexShrink: 0,
          marginTop: 3,
        }} />
        <div style={{ flex: 1, width: 2, background: "var(--border)", marginTop: 4 }} />
      </div>
      <div style={{ paddingBottom: 4 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>{label}</div>
        {detail && (
          <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 2 }}>{detail}</div>
        )}
      </div>
    </div>
  );
}

function TraceDetail({ data, onBack }) {
  const [showJson, setShowJson] = useState(false);
  const trace = data.trace ?? {};
  const answer = trace.final_answer ?? {};
  const conf = formatConfidence(answer.confidence);

  const timelineSteps = [];
  if (trace.query_analysis) {
    timelineSteps.push({
      label: "Query Analyzed",
      detail: `${trace.query_analysis.query_type} — ${trace.query_analysis.intent_summary ?? ""}`,
    });
  }
  if (trace.expanded_queries) {
    const count = trace.expanded_queries.variants?.length ?? 0;
    timelineSteps.push({ label: "Queries Expanded", detail: `${count} variant${count !== 1 ? "s" : ""} generated` });
  }
  if (trace.chunks_retrieved?.length) {
    const top = trace.chunks_retrieved[0];
    timelineSteps.push({
      label: "Retrieved",
      detail: `${trace.chunks_retrieved.length} chunks — top: ${top?.document_id ?? "—"}`,
    });
  }
  if (trace.chunks_reranked?.length) {
    const top = trace.chunks_reranked[0];
    timelineSteps.push({
      label: "Reranked",
      detail: `top: ${top?.document_id ?? "—"} (score ${top?.rerank_score?.toFixed(4) ?? "—"})`,
    });
  }
  if (trace.critic_verdict) {
    const v = trace.critic_verdict;
    timelineSteps.push({
      label: "Critic",
      detail: `${v.sufficient ? "Sufficient" : "Insufficient"} — confidence ${formatScore(v.confidence)}`,
    });
  }
  if (answer.answer) {
    timelineSteps.push({
      label: "Generated",
      detail: `confidence ${formatScore(answer.confidence)}${answer.refused ? " · refused" : ""}`,
    });
  }

  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: 10,
      padding: 24,
      marginTop: 4,
      marginBottom: 16,
    }}>
      <button
        onClick={onBack}
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          fontSize: 13,
          color: "var(--text-secondary)",
          padding: 0,
          marginBottom: 20,
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        ← Back to list
      </button>

      {/* Answer */}
      {answer.answer && (
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
            <span style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
              Answer
            </span>
            {answer.refused ? (
              <span style={{ fontSize: 12, color: "var(--text-muted)", background: "var(--surface-elevated)", border: "1px solid var(--border)", borderRadius: 999, padding: "3px 10px" }}>
                Outside knowledge base
              </span>
            ) : (
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: conf.color }} />
                <span style={{ fontSize: 13, color: conf.color }}>{conf.label}</span>
              </div>
            )}
          </div>
          <p style={{
            fontSize: 15,
            lineHeight: 1.7,
            color: answer.refused ? "var(--warning)" : "var(--text-primary)",
            fontStyle: answer.refused ? "italic" : "normal",
            whiteSpace: "pre-wrap",
          }}>
            {answer.refused ? (answer.refusal_reason || answer.answer) : answer.answer}
          </p>

          {/* Citations */}
          {answer.citations?.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>
                Sources
              </div>
              {answer.citations.map((c, i) => (
                <div key={i} style={{
                  background: "var(--surface-elevated)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  padding: "12px 14px",
                  marginTop: 6,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: c.excerpt ? 8 : 0 }}>
                    <span style={{ fontSize: 11, fontWeight: 700, background: "var(--accent)", color: "white", borderRadius: 4, padding: "1px 6px" }}>
                      {i + 1}
                    </span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>
                      {c.document_name || c.document_id}
                    </span>
                  </div>
                  {c.excerpt && (
                    <p style={{ fontSize: 12, color: "var(--text-muted)", fontStyle: "italic", lineHeight: 1.6, borderLeft: "2px solid var(--border)", paddingLeft: 10, margin: 0 }}>
                      {c.excerpt}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Vertical timeline */}
      {timelineSteps.length > 0 && (
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginBottom: 16 }}>
            Pipeline Execution
          </div>
          <div style={{ borderLeft: "2px solid var(--border)", paddingLeft: 0, marginLeft: 4 }}>
            {timelineSteps.map((s, i) => (
              <TimelineStep key={i} label={s.label} detail={s.detail} />
            ))}
          </div>
        </div>
      )}

      {/* Raw JSON toggle */}
      <div style={{ marginTop: 20, borderTop: "1px solid var(--border)", paddingTop: 16 }}>
        <button
          onClick={() => setShowJson((v) => !v)}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: 12,
            color: "var(--text-muted)",
            padding: 0,
          }}
        >
          {showJson ? "Hide raw JSON ▴" : "Show raw JSON ▾"}
        </button>
        {showJson && (
          <pre style={{
            background: "#0d0d14",
            borderRadius: 8,
            padding: 16,
            fontSize: 11,
            overflowX: "auto",
            color: "var(--text-secondary)",
            marginTop: 10,
            maxHeight: 400,
            overflowY: "auto",
          }}>
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trace row
// ---------------------------------------------------------------------------

function TraceRow({ trace, isSelected, onClick }) {
  const [hovered, setHovered] = useState(false);
  const conf = formatConfidence(trace.answer_confidence);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered || isSelected ? "var(--surface-elevated)" : "var(--surface)",
        border: `1px solid ${isSelected ? "var(--accent)" : hovered ? "var(--accent)" : "var(--border)"}`,
        borderLeft: isSelected ? "3px solid var(--accent)" : `1px solid ${hovered ? "var(--accent)" : "var(--border)"}`,
        borderRadius: 10,
        padding: "16px 20px",
        marginBottom: 8,
        cursor: "pointer",
        transition: "border-color 0.15s, background 0.15s",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)" }}>
          {truncate(trace.question, 80)}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0, marginLeft: 16 }}>
          {trace.second_pass_triggered && (
            <span style={{ fontSize: 11, color: "var(--warning)" }}>↺</span>
          )}
          <span style={{ fontSize: 12, color: "var(--text-muted)", whiteSpace: "nowrap" }}>
            {formatLatency(trace.total_latency_ms)}
          </span>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
          {formatTimestamp(trace.timestamp)}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: conf.color }} />
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{conf.label}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function TracesPage() {
  const [traces, setTraces] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedTrace, setSelectedTrace] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");

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

  useEffect(() => { refresh(); }, [refresh]);

  async function handleSelect(traceId) {
    if (selectedId === traceId) {
      setSelectedId(null);
      setSelectedTrace(null);
      return;
    }
    setSelectedId(traceId);
    setSelectedTrace(null);
    setDetailLoading(true);
    try {
      const data = await getTrace(traceId);
      setSelectedTrace(data);
    } catch (err) {
      setError(err.message ?? "Failed to load trace");
    } finally {
      setDetailLoading(false);
    }
  }

  const filtered = traces.filter((t) =>
    t.question.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 32 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 600, color: "var(--text-primary)" }}>
            Query History
          </h1>
          <p style={{ fontSize: 14, color: "var(--text-secondary)", marginTop: 6 }}>
            Browse past queries and inspect pipeline execution details.
          </p>
        </div>
        <button
          onClick={refresh}
          disabled={loading}
          style={{
            background: "none",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "8px 16px",
            fontSize: 13,
            color: "var(--text-secondary)",
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.5 : 1,
          }}
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {/* Search */}
      <input
        type="text"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search queries..."
        style={{
          width: "100%",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 10,
          padding: "10px 14px",
          fontSize: 14,
          color: "var(--text-primary)",
          outline: "none",
          marginBottom: 24,
          transition: "border-color 0.15s, box-shadow 0.15s",
        }}
        onFocus={(e) => {
          e.target.style.borderColor = "var(--accent)";
          e.target.style.boxShadow = "0 0 0 3px rgba(99,102,241,0.15)";
        }}
        onBlur={(e) => {
          e.target.style.borderColor = "var(--border)";
          e.target.style.boxShadow = "none";
        }}
      />

      {error && (
        <div style={{
          background: "rgba(239,68,68,0.08)",
          border: "1px solid rgba(239,68,68,0.2)",
          borderRadius: 8,
          padding: "10px 14px",
          fontSize: 13,
          color: "var(--danger)",
          marginBottom: 16,
        }}>
          {error}
        </div>
      )}

      {loading && (
        <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Loading traces…</p>
      )}

      {!loading && filtered.length === 0 && (
        <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
          {traces.length === 0 ? "No traces yet. Run a query to get started." : "No traces match your search."}
        </p>
      )}

      {/* Trace list */}
      {filtered.map((trace) => (
        <div key={trace.trace_id}>
          <TraceRow
            trace={trace}
            isSelected={selectedId === trace.trace_id}
            onClick={() => handleSelect(trace.trace_id)}
          />
          {selectedId === trace.trace_id && (
            detailLoading ? (
              <div style={{ padding: "16px 20px", fontSize: 13, color: "var(--text-muted)", marginBottom: 16 }}>
                Loading trace…
              </div>
            ) : selectedTrace ? (
              <TraceDetail
                data={selectedTrace}
                onBack={() => { setSelectedId(null); setSelectedTrace(null); }}
              />
            ) : null
          )}
        </div>
      ))}
    </div>
  );
}
