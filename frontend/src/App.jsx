import { NavLink, Routes, Route, Navigate } from "react-router-dom";
import QueryPage from "./pages/QueryPage";
import EvalPage from "./pages/EvalPage";
import TracesPage from "./pages/TracesPage";

function IconSearch() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <circle cx="7" cy="7" r="4.5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M10.5 10.5L13.5 13.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconBarChart() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <rect x="2" y="9" width="3" height="5" rx="1" stroke="currentColor" strokeWidth="1.5" />
      <rect x="6.5" y="5" width="3" height="9" rx="1" stroke="currentColor" strokeWidth="1.5" />
      <rect x="11" y="2" width="3" height="12" rx="1" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function IconClock() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="5.5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 5V8L10 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const NAV_ITEMS = [
  { to: "/query", label: "Query", Icon: IconSearch },
  { to: "/evaluation", label: "Evaluation", Icon: IconBarChart },
  { to: "/traces", label: "Traces", Icon: IconClock },
];

const sidebarStyle = {
  position: "fixed",
  top: 0,
  left: 0,
  width: 220,
  height: "100vh",
  background: "var(--surface)",
  borderRight: "1px solid var(--border)",
  display: "flex",
  flexDirection: "column",
  zIndex: 100,
};

const logoBlockStyle = {
  padding: "24px",
  display: "flex",
  alignItems: "center",
  gap: 12,
};

const logoMarkStyle = {
  width: 16,
  height: 16,
  background: "var(--accent)",
  borderRadius: 4,
  flexShrink: 0,
};

const navStyle = {
  flex: 1,
  padding: "8px 0",
};

const sidebarFooterStyle = {
  position: "absolute",
  bottom: 0,
  left: 0,
  right: 0,
  padding: "20px",
  fontSize: 11,
  color: "var(--text-muted)",
};

const mainStyle = {
  marginLeft: 220,
  minHeight: "100vh",
  padding: "48px",
};

const innerStyle = {
  maxWidth: 860,
};

export default function App() {
  return (
    <>
      <aside style={sidebarStyle}>
        <div style={logoBlockStyle}>
          <div style={logoMarkStyle} />
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)" }}>
              BuildCore
            </div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 1 }}>
              Intelligence
            </div>
          </div>
        </div>

        <nav style={navStyle}>
          {NAV_ITEMS.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              style={({ isActive }) => ({
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "12px 20px",
                fontSize: 14,
                fontWeight: 500,
                textDecoration: "none",
                color: isActive ? "var(--accent-light)" : "var(--text-secondary)",
                background: isActive ? "rgba(99,102,241,0.06)" : "transparent",
                borderLeft: isActive ? "2px solid var(--accent)" : "2px solid transparent",
                transition: "color 0.15s, background 0.15s",
              })}
            >
              <Icon />
              {label}
            </NavLink>
          ))}
        </nav>

        <div style={sidebarFooterStyle}>Powered by GPT-4o</div>
      </aside>

      <main style={mainStyle}>
        <div style={innerStyle}>
          <Routes>
            <Route path="/" element={<Navigate to="/query" replace />} />
            <Route path="/query" element={<QueryPage />} />
            <Route path="/evaluation" element={<EvalPage />} />
            <Route path="/traces" element={<TracesPage />} />
          </Routes>
        </div>
      </main>
    </>
  );
}
