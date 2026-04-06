import { labelFromKey, formatScore } from "../utils/formatters";

/**
 * Renders the live pipeline step timeline for a query run.
 *
 * Props:
 *   - steps: array of { step, payload } from useSSE
 *   - isStreaming: bool
 */
export default function PipelineTrace({ steps, isStreaming }) {
  if (steps.length === 0 && !isStreaming) return null;

  return (
    <div className="space-y-2">
      {steps.map(({ step, payload }, idx) => (
        <StepCard key={idx} step={step} payload={payload} />
      ))}
      {isStreaming && (
        <div className="flex items-center gap-2 text-gray-500 text-sm px-1">
          <span className="animate-pulse">●</span>
          <span>Running pipeline…</span>
        </div>
      )}
    </div>
  );
}

function StepCard({ step, payload }) {
  const meta = STEP_META[step] ?? { label: labelFromKey(step), colour: "gray" };
  const colour = COLOURS[meta.colour] ?? COLOURS.gray;

  return (
    <div className={`rounded-lg border ${colour.border} ${colour.bg} px-4 py-3`}>
      <div className="flex items-center gap-2 mb-1">
        <span className={`text-xs font-semibold uppercase tracking-wide ${colour.text}`}>
          {meta.label}
        </span>
      </div>
      <StepBody step={step} payload={payload} />
    </div>
  );
}

function StepBody({ step, payload }) {
  if (!payload) return null;

  switch (step) {
    case "query_analyzed":
      return (
        <div className="text-sm text-gray-300 space-y-1">
          <p>
            <span className="text-gray-500">Type:</span>{" "}
            <span className="font-medium">{labelFromKey(payload.query_type ?? "")}</span>
            {payload.requires_multi_hop && (
              <span className="ml-2 text-xs bg-purple-800 text-purple-200 px-1.5 py-0.5 rounded">
                multi-hop
              </span>
            )}
          </p>
          {payload.intent_summary && (
            <p className="text-gray-400 text-xs">{payload.intent_summary}</p>
          )}
          {payload.retrieval_strategy && (
            <p className="text-gray-500 text-xs font-mono">{payload.retrieval_strategy}</p>
          )}
        </div>
      );

    case "queries_expanded":
      return (
        <div className="text-sm text-gray-300 space-y-1">
          <p className="text-gray-400 text-xs">
            {(payload.variants ?? []).length} variants generated
          </p>
          <ul className="list-disc list-inside space-y-0.5 text-xs text-gray-400">
            {(payload.variants ?? []).map((v, i) => (
              <li key={i}>{v}</li>
            ))}
          </ul>
        </div>
      );

    case "chunks_retrieved":
      return (
        <ChunkList
          chunks={payload}
          scoreKey="dense_score"
          label="dense"
        />
      );

    case "chunks_reranked":
      return (
        <ChunkList
          chunks={payload}
          scoreKey="rerank_score"
          label="rerank"
        />
      );

    case "critic_verdict":
      return (
        <div className="text-sm text-gray-300 space-y-1">
          <p>
            <span
              className={`text-xs font-semibold ${
                payload.sufficient ? "text-green-400" : "text-yellow-400"
              }`}
            >
              {payload.sufficient ? "Sufficient" : "Insufficient"}
            </span>
            {payload.confidence != null && (
              <span className="ml-2 text-gray-500 text-xs">
                confidence {formatScore(payload.confidence)}
              </span>
            )}
          </p>
          {payload.reasoning && (
            <p className="text-xs text-gray-400">{payload.reasoning}</p>
          )}
          {payload.refined_query && (
            <p className="text-xs text-gray-500 font-mono">
              Refined: {payload.refined_query}
            </p>
          )}
        </div>
      );

    case "second_pass_triggered":
      return (
        <p className="text-xs text-yellow-300">
          Second retrieval pass with refined query: "
          {payload.refined_query}"
        </p>
      );

    default:
      return null;
  }
}

function ChunkList({ chunks, scoreKey, label }) {
  if (!Array.isArray(chunks) || chunks.length === 0) return null;
  return (
    <div className="text-xs text-gray-400 space-y-0.5 mt-1">
      <p className="text-gray-500 mb-1">{chunks.length} chunks</p>
      {chunks.slice(0, 5).map((c, i) => (
        <div key={i} className="flex items-center gap-2 font-mono">
          <span className="text-gray-600 w-4 text-right">{i + 1}.</span>
          <span className="truncate flex-1">{c.document_id ?? c.chunk_id}</span>
          {c[scoreKey] != null && (
            <span className="text-gray-500 flex-shrink-0">
              {label} {c[scoreKey].toFixed(3)}
            </span>
          )}
        </div>
      ))}
      {chunks.length > 5 && (
        <p className="text-gray-600">+ {chunks.length - 5} more</p>
      )}
    </div>
  );
}

const STEP_META = {
  query_analyzed: { label: "Query Analysis", colour: "blue" },
  queries_expanded: { label: "Query Expansion", colour: "indigo" },
  chunks_retrieved: { label: "Hybrid Retrieval", colour: "cyan" },
  chunks_reranked: { label: "Reranking", colour: "teal" },
  critic_verdict: { label: "Retrieval Critic", colour: "yellow" },
  second_pass_triggered: { label: "Second Pass", colour: "orange" },
  answer_generated: { label: "Answer Generated", colour: "green" },
};

const COLOURS = {
  blue: {
    border: "border-blue-800",
    bg: "bg-blue-950/40",
    text: "text-blue-400",
  },
  indigo: {
    border: "border-indigo-800",
    bg: "bg-indigo-950/40",
    text: "text-indigo-400",
  },
  cyan: {
    border: "border-cyan-800",
    bg: "bg-cyan-950/40",
    text: "text-cyan-400",
  },
  teal: {
    border: "border-teal-800",
    bg: "bg-teal-950/40",
    text: "text-teal-400",
  },
  yellow: {
    border: "border-yellow-800",
    bg: "bg-yellow-950/40",
    text: "text-yellow-400",
  },
  orange: {
    border: "border-orange-800",
    bg: "bg-orange-950/40",
    text: "text-orange-400",
  },
  green: {
    border: "border-green-800",
    bg: "bg-green-950/40",
    text: "text-green-400",
  },
  gray: {
    border: "border-gray-700",
    bg: "bg-gray-800/40",
    text: "text-gray-400",
  },
};
