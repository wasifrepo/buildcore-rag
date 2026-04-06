/**
 * Format a latency value in milliseconds to a human-readable string.
 * @param {number} ms
 * @returns {string}
 */
export function formatLatency(ms) {
  if (ms == null) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/**
 * Format a float score (0–1) as a percentage string.
 * @param {number} score
 * @returns {string}
 */
export function formatScore(score) {
  if (score == null) return "—";
  return `${Math.round(score * 100)}%`;
}

/**
 * Format an ISO-8601 UTC timestamp to a short local date+time string.
 * @param {string} iso
 * @returns {string}
 */
export function formatTimestamp(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Truncate a string to maxLen characters, appending "…" if cut.
 * @param {string} text
 * @param {number} maxLen
 * @returns {string}
 */
export function truncate(text, maxLen = 120) {
  if (!text) return "";
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + "…";
}

/**
 * Convert a snake_case or kebab-case identifier to Title Case label.
 * @param {string} key
 * @returns {string}
 */
export function labelFromKey(key) {
  return key
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Return a Tailwind colour class for a 0–1 confidence / score value.
 * @param {number} score
 * @returns {string}
 */
export function scoreColour(score) {
  if (score >= 0.8) return "text-green-400";
  if (score >= 0.6) return "text-yellow-400";
  return "text-red-400";
}
