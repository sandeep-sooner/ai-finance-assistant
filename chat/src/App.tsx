import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import "@fontsource-variable/inter";
import { request } from "./api/client";
import ChatTab from "./chat/ChatTab";
import GoalsTab from "./goals/GoalsTab";
import { useApi } from "./hooks/useApi";
import MarketTab from "./market/MarketTab";
import PortfolioTab from "./portfolio/PortfolioTab";
import SystemTab from "./system/SystemTab";
import "./shell.css";
import "./theme.css";

const svgProps = { viewBox: "0 0 24 24", "aria-hidden": true, focusable: false } as const;

const ChatIcon = () => (
  <svg {...svgProps}>
    <path d="M21 12a8 8 0 0 1-11.5 7.2L4 20l1-4.5A8 8 0 1 1 21 12z" />
  </svg>
);
const PortfolioIcon = () => (
  <svg {...svgProps}>
    <path d="M12 3a9 9 0 1 0 9 9h-9z" />
    <path d="M15 3.3A9 9 0 0 1 20.7 9H15z" />
  </svg>
);
const MarketIcon = () => (
  <svg {...svgProps}>
    <path d="M3 17l6-6 4 4 8-8" />
    <path d="M15 7h6v6" />
  </svg>
);
const GoalsIcon = () => (
  <svg {...svgProps}>
    <circle cx="12" cy="12" r="9" />
    <circle cx="12" cy="12" r="4.5" />
    <circle cx="12" cy="12" r="0.8" fill="currentColor" />
  </svg>
);
const SystemIcon = () => (
  <svg {...svgProps}>
    <path d="M4 19V10M10 19V5M16 19v-6M22 19H2" />
  </svg>
);
const BrandIcon = () => (
  <svg {...svgProps}>
    <path d="M5 16l4-4 3 3 6-7" />
  </svg>
);
const SunIcon = () => (
  <svg {...svgProps}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
);
const MoonIcon = () => (
  <svg {...svgProps}>
    <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />
  </svg>
);

const TABS = [
  { id: "chat", label: "Assistant", Icon: ChatIcon },
  { id: "portfolio", label: "Portfolio", Icon: PortfolioIcon },
  { id: "market", label: "Market", Icon: MarketIcon },
  { id: "goals", label: "Goals", Icon: GoalsIcon },
  { id: "system", label: "System", Icon: SystemIcon },
] as const;

type TabId = (typeof TABS)[number]["id"];
type Theme = "light" | "dark";

interface Health {
  status: string;
  llm_provider: string;
  models: string[];
  market_data_provider: string;
  rag: string;
}

const THEME_KEY = "finance-assistant-theme";

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* storage blocked */
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

const shortModel = (name: string) => name.split(":").pop() ?? name;

export default function App() {
  const [tab, setTab] = useState<TabId>("chat");
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const activeIndex = TABS.findIndex((t) => t.id === tab);

  const health = useApi<Health>("health", (signal) => request<Health>("/api/health", { signal, timeoutMs: 8000 }), {
    refreshMs: 60_000,
  });

  useEffect(() => {
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* storage blocked */
    }
  }, [theme]);

  // Arrow keys, Home and End move between tabs, as the ARIA tabs pattern expects.
  const onKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    let next: number;
    if (e.key === "ArrowDown" || e.key === "ArrowRight") next = (activeIndex + 1) % TABS.length;
    else if (e.key === "ArrowUp" || e.key === "ArrowLeft") next = (activeIndex - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    else return;
    e.preventDefault();
    setTab(TABS[next].id);
    tabRefs.current[next]?.focus();
  };

  const h = health.data;
  const state = health.error ? "down" : !h ? "idle" : h.llm_provider === "fake" ? "demo" : "ok";
  const statusText = { down: "Backend unreachable", idle: "Connecting…", demo: "Demo mode (no API key)", ok: "All systems live" }[state];

  return (
    <div className="app" data-theme={theme}>
      <aside className="app-side">
        <div className="app-brand">
          <span className="app-brand-mark">
            <BrandIcon />
          </span>
          <span className="app-brand-text">
            <span className="app-brand-name">Finance Assistant</span>
            <span className="app-brand-sub">Learn investing with AI</span>
          </span>
        </div>

        <nav className="app-nav" role="tablist" aria-label="Sections" aria-orientation="vertical" onKeyDown={onKeyDown}>
          <span className="app-nav-label">Workspace</span>
          {TABS.map(({ id, label, Icon }, i) => (
            <button
              key={id}
              ref={(el) => {
                tabRefs.current[i] = el;
              }}
              type="button"
              role="tab"
              id={`tab-${id}`}
              aria-selected={tab === id}
              aria-controls={`pane-${id}`}
              tabIndex={tab === id ? 0 : -1}
              className="app-tab"
              onClick={() => setTab(id)}
            >
              <Icon />
              {label}
            </button>
          ))}
        </nav>

        <div className="app-status" aria-live="polite">
          <div className="app-status-head">
            <span className="app-dot" data-state={state} />
            {statusText}
          </div>
          {h && (
            <dl>
              <dt>Fast model</dt>
              <dd title={h.models[0]}>{shortModel(h.models[0] ?? "—")}</dd>
              <dt>Smart model</dt>
              <dd title={h.models[1] ?? h.models[0]}>{shortModel(h.models[1] ?? h.models[0] ?? "—")}</dd>
              <dt>Market data</dt>
              <dd>{h.market_data_provider}</dd>
              <dt>Knowledge base</dt>
              <dd>{h.rag === "NullRetriever" ? "off" : "indexed"}</dd>
            </dl>
          )}
        </div>

        <button
          type="button"
          className="app-theme"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
          <span>{theme === "dark" ? "Light theme" : "Dark theme"}</span>
        </button>
      </aside>

      <main className="app-main">
        {/* Every pane stays mounted and is only hidden, so the chat keeps its
            conversation and any in-flight stream while you visit other tabs. */}
        <div className="app-pane" role="tabpanel" id="pane-chat" aria-labelledby="tab-chat" hidden={tab !== "chat"}>
          <ChatTab />
        </div>
        <div className="app-pane" role="tabpanel" id="pane-portfolio" aria-labelledby="tab-portfolio" hidden={tab !== "portfolio"}>
          <PortfolioTab />
        </div>
        <div className="app-pane" role="tabpanel" id="pane-market" aria-labelledby="tab-market" hidden={tab !== "market"}>
          {/* Only polls for fresh prices while visible. */}
          <MarketTab active={tab === "market"} />
        </div>
        <div className="app-pane" role="tabpanel" id="pane-goals" aria-labelledby="tab-goals" hidden={tab !== "goals"}>
          <GoalsTab />
        </div>
        <div className="app-pane" role="tabpanel" id="pane-system" aria-labelledby="tab-system" hidden={tab !== "system"}>
          <SystemTab active={tab === "system"} />
        </div>
      </main>
    </div>
  );
}
