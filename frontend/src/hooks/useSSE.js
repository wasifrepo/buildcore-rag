import { useState, useRef, useCallback } from "react";

/**
 * Generic hook for consuming a Server-Sent Events stream from a POST endpoint.
 *
 * The stream format expected by this hook:
 *   data: {"step": "<event_name>", "payload": { ... }}\n\n
 *
 * Returns:
 *   - steps: array of { step, payload } received so far
 *   - isStreaming: true while the stream is open
 *   - error: error message string or null
 *   - start(fetchFn): opens the stream; fetchFn must return a Promise<Response>
 *   - reset(): clears all state
 */
export function useSSE() {
  const [steps, setSteps] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const readerRef = useRef(null);

  const reset = useCallback(() => {
    if (readerRef.current) {
      readerRef.current.cancel();
      readerRef.current = null;
    }
    setSteps([]);
    setIsStreaming(false);
    setError(null);
  }, []);

  /**
   * Start consuming an SSE stream.
   * @param {() => Promise<Response>} fetchFn — zero-argument function that
   *   returns a fetch Response whose body is the SSE stream.
   * @param {(step: string, payload: object) => void} [onStep] — optional
   *   callback invoked for each parsed event.
   * @param {() => void} [onDone] — optional callback invoked when stream ends.
   */
  const start = useCallback(async (fetchFn, onStep, onDone) => {
    reset();
    setIsStreaming(true);

    try {
      const res = await fetchFn();
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const reader = res.body.getReader();
      readerRef.current = reader;
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Split on double-newline (SSE message boundary)
        const messages = buffer.split("\n\n");
        // Last element may be incomplete — keep it in the buffer
        buffer = messages.pop() ?? "";

        for (const message of messages) {
          const dataLine = message
            .split("\n")
            .find((l) => l.startsWith("data: "));
          if (!dataLine) continue;

          try {
            const { step, payload } = JSON.parse(dataLine.slice(6));
            setSteps((prev) => [...prev, { step, payload }]);
            onStep?.(step, payload);
          } catch {
            // Skip malformed events
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        setError(err.message ?? "Stream error");
      }
    } finally {
      readerRef.current = null;
      setIsStreaming(false);
      onDone?.();
    }
  }, [reset]);

  return { steps, isStreaming, error, start, reset };
}
