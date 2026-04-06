import { useState } from "react";
import { useSSE } from "../hooks/useSSE";
import QueryPanel from "../components/QueryPanel";
import PipelineTrace from "../components/PipelineTrace";
import AnswerCard from "../components/AnswerCard";

export default function QueryPage() {
  const { steps, isStreaming, error, start, reset } = useSSE();
  const [answerPayload, setAnswerPayload] = useState(null);

  async function handleSubmit(question) {
    reset();
    setAnswerPayload(null);

    await start(
      () =>
        fetch("/query/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        }),
      (step, payload) => {
        if (step === "answer_generated") {
          setAnswerPayload(payload);
        }
      }
    );
  }

  // Exclude answer_generated from the timeline — shown separately below
  const timelineSteps = steps.filter((s) => s.step !== "answer_generated");

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-100">Query Pipeline</h1>
        <p className="text-sm text-gray-500 mt-1">
          Ask a question — watch each retrieval stage complete in real time.
        </p>
      </div>

      <QueryPanel onSubmit={handleSubmit} disabled={isStreaming} />

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {timelineSteps.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Pipeline Steps
          </p>
          <PipelineTrace steps={timelineSteps} isStreaming={isStreaming} />
        </div>
      )}

      {answerPayload && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Final Answer
          </p>
          <AnswerCard payload={answerPayload} />
        </div>
      )}
    </div>
  );
}
