import { useState } from "react";

/**
 * Query input form with submit button.
 *
 * Props:
 *   - onSubmit(question: string): called when the form is submitted
 *   - disabled: bool — disables input and button while streaming
 */
export default function QueryPanel({ onSubmit, disabled }) {
  const [question, setQuestion] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || disabled) return;
    onSubmit(q);
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-3">
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        disabled={disabled}
        placeholder="Ask a question about BuildCore Operations…"
        className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={disabled || !question.trim()}
        className="px-5 py-2.5 bg-orange-500 hover:bg-orange-600 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded-lg transition-colors"
      >
        {disabled ? "Running…" : "Ask"}
      </button>
    </form>
  );
}
