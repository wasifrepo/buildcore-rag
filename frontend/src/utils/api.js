export const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Opens an SSE connection to the query stream endpoint.
 * Calls onStep(step, payload) for each pipeline step received.
 * Calls onDone() when the stream closes.
 * Calls onError(err) on failure.
 */
export function streamQuery(question, { onStep, onDone, onError }) {
  const url = `${BASE_URL}/query/stream`;

  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  })
    .then((res) => {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      function read() {
        reader.read().then(({ done, value }) => {
          if (done) {
            onDone?.();
            return;
          }
          const text = decoder.decode(value);
          const lines = text.split("\n").filter((l) => l.startsWith("data: "));
          for (const line of lines) {
            try {
              const data = JSON.parse(line.replace("data: ", ""));
              onStep?.(data.step, data.payload);
            } catch {
              // malformed chunk, skip
            }
          }
          read();
        });
      }

      read();
    })
    .catch(onError);
}

export async function runEvaluation() {
  const res = await fetch(`${BASE_URL}/evaluate/run`, { method: "POST" });
  return res.body;
}

export async function listTraces() {
  const res = await fetch(`${BASE_URL}/traces/`);
  return res.json();
}

export async function getTrace(traceId) {
  const res = await fetch(`${BASE_URL}/traces/${traceId}`);
  return res.json();
}

export async function getLatestEval() {
  const res = await fetch(`${BASE_URL}/evaluate/latest`);
  if (res.status === 404) return null;
  return res.json();
}
