import { NavLink, Routes, Route, Navigate } from "react-router-dom";
import QueryPage from "./pages/QueryPage";
import EvalPage from "./pages/EvalPage";
import TracesPage from "./pages/TracesPage";

const NAV_ITEMS = [
  { to: "/query", label: "Query", icon: "🔍" },
  { to: "/evaluation", label: "Evaluation", icon: "📊" },
  { to: "/traces", label: "Traces", icon: "🗂️" },
];

export default function App() {
  return (
    <div className="flex h-screen bg-gray-950 text-gray-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-gray-800">
          <div className="text-orange-500 font-bold text-lg leading-tight">
            BuildCore
          </div>
          <div className="text-gray-500 text-xs mt-0.5">RAG Pipeline</div>
        </div>

        {/* Nav links */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-orange-500/20 text-orange-400"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
                }`
              }
            >
              <span className="text-base">{icon}</span>
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-gray-800 text-xs text-gray-600">
          BuildCore Operations
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/query" replace />} />
          <Route path="/query" element={<QueryPage />} />
          <Route path="/evaluation" element={<EvalPage />} />
          <Route path="/traces" element={<TracesPage />} />
        </Routes>
      </main>
    </div>
  );
}
