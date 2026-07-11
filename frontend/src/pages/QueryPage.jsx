import { useState } from "react";
import { useSSE } from "../hooks/useSSE";
import { BASE_URL } from "../utils/api";
import { formatConfidence, formatScore, truncate } from "../utils/formatters";

// ---------------------------------------------------------------------------
// Pipeline step definitions
// ---------------------------------------------------------------------------

const PIPELINE_STEPS = [
  { label: "Analyzing", triggers: ["query_analyzed"] },
  { label: "Retrieving", triggers: ["queries_expanded", "chunks_retrieved"] },
  { label: "Ranking", triggers: ["chunks_reranked"] },
  { label: "Verifying", triggers: ["critic_verdict"] },
  { label: "Answering", triggers: ["answer_generated"] },
];

function getStepStatus(stepIndex, completedTriggers, isStreaming) {
  const triggers = PIPELINE_STEPS[stepIndex].triggers;
  const done = triggers.some((t) => completedTriggers.has(t));
  if (done) return "complete";
  // Active = all previous steps done and streaming is still open
  if (isStreaming) {
    const prevDone = stepIndex === 0 || PIPELINE_STEPS[stepIndex - 1].triggers.some(
      (t) => completedTriggers.has(t)
    );
    if (prevDone) return "active";
  }
  return "pending";
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function PipelineProgressBar({ completedTriggers, isStreaming }) {
  return (
    <div style={{ display: "flex", alignItems: "center", marginTop: 32 }}>
      {PIPELINE_STEPS.map((step, i) => {
        const status = getStepStatus(i, completedTriggers, isStreaming);
        const isLast = i === PIPELINE_STEPS.length - 1;
        return (
          <div key={step.label} style={{ display: "flex", alignItems: "center", flex: isLast ? "0 0 auto" : 1 }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
              <StepCircle status={status} />
              <span style={{
                fontSize: 11,
                color: status === "pending" ? "var(--text-muted)" : "var(--text-primary)",
                whiteSpace: "nowrap",
              }}>
                {step.label}
              </span>
            </div>
            {!isLast && (
              <div style={{
                flex: 1,
                height: 1,
                background: status === "complete" ? "var(--accent)" : "var(--border)",
                margin: "0 8px",
                marginBottom: 18,
                transition: "background 0.3s",
              }} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function StepCircle({ status }) {
  const base = {
    width: 20,
    height: 20,
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "all 0.3s",
    flexShrink: 0,
  };

  if (status === "complete") {
    return (
      <div style={{ ...base, background: "var(--accent)", border: "2px solid var(--accent)" }}>
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
          <path d="M2 5L4 7L8 3" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    );
  }
  if (status === "active") {
    return (
      <div style={{ ...base, border: "2px solid var(--accent)", background: "transparent", position: "relative" }}>
        <div style={{
          position: "absolute",
          width: 28,
          height: 28,
          borderRadius: "50%",
          border: "2px solid var(--accent)",
          opacity: 0.3,
          animation: "pulse 1.5s ease-in-out infinite",
        }} />
      </div>
    );
  }
  return <div style={{ ...base, border: "2px solid var(--border)", background: "transparent" }} />;
}

function PipelineDetailsPanel({ stepsData }) {
  const queryAnalysis = stepsData.query_analyzed;
  const expandedQueries = stepsData.queries_expanded;
  const retrievedChunks = stepsData.chunks_retrieved;
  const rerankedChunks = stepsData.chunks_reranked;

  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: 8,
      padding: 16,
      marginTop: 12,
      display: "flex",
      flexDirection: "column",
      gap: 16,
    }}>
      {queryAnalysis && (
        <section>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>
            Query Analysis
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <span style={{
              fontSize: 11,
              fontWeight: 600,
              background: "rgba(99,102,241,0.15)",
              color: "var(--accent-light)",
              borderRadius: 999,
              padding: "2px 8px",
            }}>
              {queryAnalysis.query_type}
            </span>
            {queryAnalysis.requires_multi_hop && (
              <span style={{
                fontSize: 11,
                fontWeight: 600,
                background: "rgba(245,158,11,0.15)",
                color: "var(--warning)",
                borderRadius: 999,
                padding: "2px 8px",
              }}>
                multi-hop
              </span>
            )}
          </div>
          {queryAnalysis.intent_summary && (
            <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 4 }}>
              {queryAnalysis.intent_summary}
            </p>
          )}
          {queryAnalysis.retrieval_strategy && (
            <code style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "monospace" }}>
              {queryAnalysis.retrieval_strategy}
            </code>
          )}
        </section>
      )}

      {expandedQueries?.variants?.length > 0 && (
        <section>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>
            Expanded Queries
          </div>
          <ol style={{ paddingLeft: 16, display: "flex", flexDirection: "column", gap: 4 }}>
            {expandedQueries.variants.map((v, i) => (
              <li key={i} style={{ fontSize: 13, color: "var(--text-secondary)" }}>{v}</li>
            ))}
          </ol>
        </section>
      )}

      {retrievedChunks?.length > 0 && (
        <ChunkTable title="Retrieved Chunks" chunks={retrievedChunks} scoreKey="dense_score" />
      )}

      {rerankedChunks?.length > 0 && (
        <ChunkTable title="Reranked Chunks" chunks={rerankedChunks} scoreKey="rerank_score" />
      )}
    </div>
  );
}

function ChunkTable({ title, chunks, scoreKey }) {
  const top5 = chunks.slice(0, 5);
  return (
    <section>
      <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>
        {title}
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--border)" }}>
            {["Rank", "Document", "Type", "Score"].map((h) => (
              <th key={h} style={{ textAlign: "left", padding: "4px 8px", color: "var(--text-muted)", fontWeight: 600, fontSize: 11 }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {top5.map((c, i) => (
            <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "6px 8px", color: "var(--text-muted)" }}>{i + 1}</td>
              <td style={{ padding: "6px 8px", color: "var(--text-secondary)", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {c.document_id}
              </td>
              <td style={{ padding: "6px 8px", color: "var(--text-muted)" }}>{c.document_type}</td>
              <td style={{ padding: "6px 8px", color: "var(--accent-light)", fontVariantNumeric: "tabular-nums" }}>
                {c[scoreKey] != null ? c[scoreKey].toFixed(4) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function AnswerCard({ payload }) {
  const { answer, citations = [], confidence, refused, refusal_reason } = payload;
  const conf = formatConfidence(confidence);

  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: 12,
      padding: 28,
      marginTop: 28,
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <span style={{ fontSize: 13, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Answer
        </span>
        {refused ? (
          <span style={{
            fontSize: 12,
            color: "var(--text-muted)",
            background: "var(--surface-elevated)",
            border: "1px solid var(--border)",
            borderRadius: 999,
            padding: "3px 10px",
          }}>
            Outside knowledge base
          </span>
        ) : (
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: conf.color }} />
            <span style={{ fontSize: 13, color: conf.color }}>{conf.label}</span>
          </div>
        )}
      </div>

      {refused ? (
        <p style={{ fontSize: 15, lineHeight: 1.7, color: "var(--warning)", fontStyle: "italic" }}>
          {refusal_reason || answer}
        </p>
      ) : (
        <p style={{ fontSize: 15, lineHeight: 1.7, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
          {answer}
        </p>
      )}

      {citations.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>
            Sources
          </div>
          {citations.map((c, i) => (
            <div key={i} style={{
              background: "var(--surface-elevated)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: "14px 16px",
              marginTop: 8,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: c.excerpt ? 8 : 0 }}>
                <span style={{
                  fontSize: 11,
                  fontWeight: 700,
                  background: "var(--accent)",
                  color: "white",
                  borderRadius: 4,
                  padding: "1px 6px",
                }}>
                  {i + 1}
                </span>
                <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>
                  {c.document_name || c.document_id}
                </span>
              </div>
              {c.excerpt && (
                <p style={{
                  fontSize: 12,
                  color: "var(--text-muted)",
                  fontStyle: "italic",
                  lineHeight: 1.6,
                  borderLeft: "2px solid var(--border)",
                  paddingLeft: 10,
                  margin: 0,
                }}>
                  {c.excerpt}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function QueryPage() {
  const { steps, isStreaming, error, start, reset } = useSSE();
  const [question, setQuestion] = useState("");
  const [showDetails, setShowDetails] = useState(false);

  // Extract per-step payloads
  const stepsData = {};
  for (const { step, payload } of steps) {
    stepsData[step] = payload;
  }

  const completedTriggers = new Set(steps.map((s) => s.step));
  const answerPayload = stepsData["answer_generated"];
  const criticVerdict = stepsData["critic_verdict"];
  const secondPassPayload = stepsData["second_pass_triggered"];
  const pipelineError = stepsData["error"];

  const hasActivity = steps.length > 0 || isStreaming;

  async function handleSubmit(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || isStreaming) return;
    reset();
    await start(() =>
      fetch(`${BASE_URL}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      })
    );
  }

  return (
    <div>
      {/* Pulse animation keyframes */}
      <style>{`
        @keyframes pulse {
          0%, 100% { transform: scale(1); opacity: 0.3; }
          50% { transform: scale(1.4); opacity: 0; }
        }
      `}</style>

      {/* Page header */}
      <h1 style={{ fontSize: 24, fontWeight: 600, color: "var(--text-primary)" }}>
        Ask BuildCore Intelligence
      </h1>
      <p style={{ fontSize: 14, color: "var(--text-secondary)", marginTop: 6, marginBottom: 32 }}>
        Search across SOPs, contracts, incident reports, maintenance manuals, and compliance checklists.
      </p>

      {/* Search bar */}
      <form onSubmit={handleSubmit} style={{ display: "flex" }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={isStreaming}
          placeholder="What does the fall protection SOP say about harness inspection?"
          style={{
            flex: 1,
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            padding: "14px 18px",
            fontSize: 15,
            color: "var(--text-primary)",
            outline: "none",
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
        <button
          type="submit"
          disabled={isStreaming || !question.trim()}
          style={{
            background: "var(--accent)",
            color: "white",
            border: "none",
            borderRadius: 8,
            padding: "14px 24px",
            fontSize: 14,
            fontWeight: 600,
            cursor: isStreaming || !question.trim() ? "not-allowed" : "pointer",
            marginLeft: 12,
            whiteSpace: "nowrap",
            opacity: isStreaming || !question.trim() ? 0.5 : 1,
            transition: "opacity 0.15s",
          }}
        >
          Ask →
        </button>
      </form>

      {(error || pipelineError) && (
        <div style={{
          marginTop: 16,
          background: "rgba(239,68,68,0.08)",
          border: "1px solid rgba(239,68,68,0.2)",
          borderRadius: 8,
          padding: "10px 14px",
          fontSize: 13,
          color: "var(--danger)",
        }}>
          {error || `${pipelineError.type}: ${pipelineError.message}`}
        </div>
      )}

      {/* Pipeline progress bar */}
      {hasActivity && (
        <PipelineProgressBar
          completedTriggers={completedTriggers}
          isStreaming={isStreaming}
        />
      )}

      {/* Second pass banner */}
      {secondPassPayload && (
        <div style={{
          background: "rgba(245,158,11,0.08)",
          border: "1px solid rgba(245,158,11,0.2)",
          borderRadius: 8,
          padding: "10px 14px",
          marginTop: 16,
          fontSize: 13,
          color: "var(--warning)",
        }}>
          ⟳ Second retrieval pass triggered — {secondPassPayload.refined_query}
        </div>
      )}

      {/* Pipeline details disclosure */}
      {hasActivity && (
        <div style={{ marginTop: 16 }}>
          <button
            onClick={() => setShowDetails((v) => !v)}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              fontSize: 12,
              color: "var(--text-muted)",
              padding: 0,
            }}
          >
            {showDetails ? "Hide pipeline details ▴" : "View pipeline details ▾"}
          </button>
          {showDetails && <PipelineDetailsPanel stepsData={stepsData} />}
        </div>
      )}

      {/* Answer card */}
      {answerPayload && <AnswerCard payload={answerPayload} />}
    </div>
  );
}
