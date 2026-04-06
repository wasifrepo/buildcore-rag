import { formatScore, scoreColour } from "../utils/formatters";

/**
 * Displays the final generated answer with citations and confidence.
 *
 * Props:
 *   - payload: the answer_generated SSE payload
 *     { answer, citations, confidence, refused, refusal_reason }
 */
export default function AnswerCard({ payload }) {
  if (!payload) return null;

  const { answer, citations = [], confidence, refused, refusal_reason } = payload;

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-900 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800">
        <span className="text-sm font-semibold text-gray-300">Answer</span>
        <div className="flex items-center gap-3">
          {refused && (
            <span className="text-xs bg-red-900/60 text-red-300 border border-red-800 px-2 py-0.5 rounded">
              Refused
            </span>
          )}
          {confidence != null && (
            <span className={`text-sm font-medium ${scoreColour(confidence)}`}>
              {formatScore(confidence)} confidence
            </span>
          )}
        </div>
      </div>

      {/* Body */}
      <div className="px-5 py-4">
        {refused && refusal_reason ? (
          <p className="text-yellow-300 text-sm italic">{refusal_reason}</p>
        ) : (
          <p className="text-gray-200 text-sm leading-relaxed whitespace-pre-wrap">
            {answer}
          </p>
        )}
      </div>

      {/* Citations */}
      {citations.length > 0 && (
        <div className="px-5 pb-4">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Citations
          </p>
          <div className="space-y-2">
            {citations.map((c, i) => (
              <CitationRow key={i} citation={c} index={i + 1} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function CitationRow({ citation, index }) {
  return (
    <div className="rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-xs">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-orange-400 font-semibold">[{index}]</span>
        <span className="text-gray-300 font-medium">
          {citation.document_name ?? citation.document_id}
        </span>
        <span className="text-gray-600 font-mono text-xs ml-auto truncate max-w-xs">
          {citation.chunk_id}
        </span>
      </div>
      {citation.excerpt && (
        <p className="text-gray-400 leading-relaxed line-clamp-3">
          {citation.excerpt}
        </p>
      )}
    </div>
  );
}
