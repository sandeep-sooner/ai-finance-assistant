import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import Markdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { useChatStream } from "../hooks/useChatStream";
import { AGENT_LABELS, DISCLAIMER } from "../lib/constants";
import type { CallTrace, Level, StreamMeta } from "../types/api";
import type { ChatMessage } from "./types";
import "./chat.css";

// Links in answers open in a new tab. react-markdown does not render raw HTML, so model output stays safe.
const MD_COMPONENTS: Components = {
  a: ({ href, title, children }) => (
    <a href={href} title={title} target="_blank" rel="noreferrer">
      {children}
    </a>
  ),
};

/** One card per specialist agent: what it does, and a prompt that reaches it. */
const AGENT_CARDS = [
  { agent: "finance_qa", title: "Learn a concept", hint: "Plain-language answers with sources", prompt: "What is an ETF and how is it different from a mutual fund?" },
  { agent: "portfolio", title: "Check a portfolio", hint: "Diversification, risk and fund costs", prompt: "Analyze my portfolio: 40 AAPL, 10 MSFT, 25 VTI, 30 BND" },
  { agent: "market", title: "Look up the market", hint: "Live quotes and recent trend", prompt: "How is the stock market doing today?" },
  { agent: "goal_planning", title: "Plan a goal", hint: "Projection across three risk profiles", prompt: "I want to save $50,000 in 10 years. I have $5,000 and can add $300 a month" },
  { agent: "news", title: "Catch up on news", hint: "Headlines summarised and explained", prompt: "Any news on NVDA?" },
  { agent: "tax", title: "Understand taxes", hint: "Accounts, capital gains, harvesting", prompt: "How is a Roth IRA different from a traditional IRA?" },
] as const;

const LEVELS: { value: Level; label: string }[] = [
  { value: "beginner", label: "Beginner" },
  { value: "intermediate", label: "Intermediate" },
];

const shortModel = (name: string) => name.split(":").pop() ?? name;
const seconds = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

/* ---------- Icons (inline, no dependency) ---------- */

const svgProps = { viewBox: "0 0 24 24", "aria-hidden": true, focusable: false } as const;

const SendIcon = () => (
  <svg {...svgProps}>
    <path d="M12 19V5M5 12l7-7 7 7" />
  </svg>
);
const StopIcon = () => (
  <svg {...svgProps}>
    <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" />
  </svg>
);
const CopyIcon = () => (
  <svg {...svgProps}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V6a2 2 0 0 1 2-2h9" />
  </svg>
);
const CheckIcon = () => (
  <svg {...svgProps}>
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </svg>
);
const DownIcon = () => (
  <svg {...svgProps}>
    <path d="M12 5v14M5 12l7 7 7-7" />
  </svg>
);

/* ---------- Message pieces ---------- */

function AgentChips({ agents }: { agents?: string[] }) {
  if (!agents?.length) return null;
  return (
    <div className="fa-agents" aria-label="Answered by">
      {agents.map((agent) => (
        <span key={agent} className="fa-agent" data-agent={agent}>
          <span className="fa-agent-dot" />
          {AGENT_LABELS[agent] ?? agent}
        </span>
      ))}
    </div>
  );
}

/** How this answer was produced: model tier, cache, recovery and timing. One group per model call. */
function MetaChips({ meta }: { meta?: StreamMeta }) {
  if (!meta?.calls.length) return null;
  const describe = (c: CallTrace) =>
    [`Complexity ${c.score.toFixed(2)}`, ...c.reasons, c.fallbacks.length ? `Failed first: ${c.fallbacks.join(", ")}` : ""]
      .filter(Boolean)
      .join("\n");
  return (
    <div className="fa-meta" aria-label="How this answer was produced">
      {meta.calls.map((c, i) => (
        <span key={i} className="fa-meta-group" title={describe(c)}>
          {c.degraded ? (
            <span className="fa-chip" data-tone="warn">Offline answer</span>
          ) : (
            <span className="fa-chip" data-tone={c.tier === "smart" ? "smart" : "fast"}>
              {c.tier === "smart" ? "Smart model" : "Fast model"} · {shortModel(c.model)}
            </span>
          )}
          {c.cache_hit && <span className="fa-chip" data-tone="ok">Cached</span>}
          {!c.degraded && (c.fallbacks.length > 0 || c.continued) && (
            <span className="fa-chip" data-tone="warn">Recovered by backup</span>
          )}
          {!c.degraded && !c.fallbacks.length && c.retries > 0 && <span className="fa-chip" data-tone="warn">Retried</span>}
          <span className="fa-chip">{seconds(c.latency_ms)}</span>
        </span>
      ))}
      {meta.route.method === "follow_up" && <span className="fa-chip">Follow-up</span>}
      {meta.route.method === "llm" && <span className="fa-chip">Routed by model</span>}
    </div>
  );
}

function AssistantMessage({
  message,
  streaming,
  copied,
  onCopy,
}: {
  message: ChatMessage;
  streaming: boolean;
  copied: boolean;
  onCopy: (m: ChatMessage) => void;
}) {
  const thinking = streaming && message.content === "";
  // The backend's output guardrail may already append the disclaimer to the text.
  const showDisclaimer = !!message.disclaimer && !message.content.includes(message.disclaimer.trim());

  return (
    <article className="fa-msg fa-msg--assistant">
      <AgentChips agents={message.agents} />

      {thinking ? (
        <div className="fa-dots" role="status" aria-label="Assistant is thinking">
          <span />
          <span />
          <span />
        </div>
      ) : (
        <div className="fa-body">
          <Markdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
            {message.content}
          </Markdown>
          {streaming && <span className="fa-caret" aria-hidden="true" />}
        </div>
      )}

      {message.failed && <p className="fa-failed">This reply did not finish.</p>}

      {showDisclaimer && <p className="fa-msg-disclaimer">{message.disclaimer}</p>}

      {!!message.sources?.length && (
        <details className="fa-sources">
          <summary className="fa-sources-label">
            Sources ({message.sources.length})
          </summary>
          <ol>
            {message.sources.map((s, i) => {
              const inner = (
                <>
                  <span className="fa-source-n">{i + 1}</span>
                  <span className="fa-source-title">{s.title}</span>
                  {s.category && <span className="fa-source-cat">{s.category}</span>}
                </>
              );
              return (
                <li key={`${s.title}-${i}`}>
                  {s.url ? (
                    <a className="fa-source" href={s.url} target="_blank" rel="noreferrer">
                      {inner}
                    </a>
                  ) : (
                    <span className="fa-source">{inner}</span>
                  )}
                </li>
              );
            })}
          </ol>
        </details>
      )}

      {!streaming && <MetaChips meta={message.meta} />}

      {!streaming && message.content && (
        <div className="fa-actions">
          <button type="button" className="fa-ghost" onClick={() => onCopy(message)}>
            {copied ? <CheckIcon /> : <CopyIcon />}
            <span>{copied ? "Copied" : "Copy"}</span>
          </button>
        </div>
      )}
    </article>
  );
}

/* ---------- Main component ---------- */

export default function ChatTab() {
  const [level, setLevel] = useState<Level>("beginner");
  const { messages, isStreaming, error, send, stop, retry, reset } = useChatStream(level);

  const [input, setInput] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [showJump, setShowJump] = useState(false);

  const logRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const stickRef = useRef(true); // true while the user is reading the newest text

  // Grow the textarea with its content, up to the CSS max-height.
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

  // Follow new tokens only if the user has not scrolled up.
  useEffect(() => {
    const el = logRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const onScroll = () => {
    const el = logRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    stickRef.current = nearBottom;
    setShowJump(!nearBottom);
  };

  const jumpToLatest = () => {
    const el = logRef.current;
    if (!el) return;
    stickRef.current = true;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  const submit = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isStreaming) return;
      stickRef.current = true;
      void send(trimmed);
      setInput("");
    },
    [isStreaming, send]
  );

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    submit(input);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends, Shift+Enter adds a line. Ignore Enter while an IME is composing.
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit(input);
    }
  };

  const copyAnswer = async (m: ChatMessage) => {
    try {
      await navigator.clipboard.writeText(m.content);
      setCopiedId(m.id);
      window.setTimeout(() => setCopiedId(null), 1500);
    } catch {
      /* clipboard blocked: nothing to do */
    }
  };

  const lastIndex = messages.length - 1;
  const isEmpty = messages.length === 0;

  return (
    <section className="fa-chat" aria-label="Chat">
      <header className="fa-head">
        <div className="fa-level" role="radiogroup" aria-label="Explanation level">
          {LEVELS.map((l) => (
            <button
              key={l.value}
              type="button"
              role="radio"
              aria-checked={level === l.value}
              className="fa-level-opt"
              onClick={() => setLevel(l.value)}
            >
              {l.label}
            </button>
          ))}
        </div>
        <button type="button" className="fa-ghost" onClick={reset} disabled={isEmpty}>
          New chat
        </button>
      </header>

      <div
        ref={logRef}
        className="fa-log"
        onScroll={onScroll}
        role="log"
        aria-live="polite"
        aria-busy={isStreaming}
      >
        <div className="fa-thread">
          {isEmpty && (
            <div className="fa-empty">
              <span className="fa-eyebrow">Six specialist agents · grounded in a curated knowledge base</span>
              <h2>
                What do you want to <em>understand about money</em> today?
              </h2>
              <p>Ask a question, share your holdings, or plan a goal. Each request is routed to the right specialist.</p>
              <div className="fa-cards">
                {AGENT_CARDS.map((c) => (
                  <button key={c.agent} type="button" className="fa-card" data-agent={c.agent} onClick={() => submit(c.prompt)}>
                    <span className="fa-card-head">
                      <span className="fa-card-dot" />
                      {c.title}
                    </span>
                    <span className="fa-card-hint">{c.hint}</span>
                    <span className="fa-card-prompt">“{c.prompt}”</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <article key={m.id} className="fa-msg fa-msg--user">
                {m.content}
              </article>
            ) : (
              <AssistantMessage
                key={m.id}
                message={m}
                streaming={isStreaming && i === lastIndex}
                copied={copiedId === m.id}
                onCopy={copyAnswer}
              />
            )
          )}
        </div>
      </div>

      <div className="fa-dock">
        {showJump && (
          <button type="button" className="fa-jump" onClick={jumpToLatest} aria-label="Jump to latest">
            <DownIcon />
          </button>
        )}

        {error && (
          <div className="fa-error" role="alert">
            <span>{error}</span>
            <button type="button" onClick={retry}>
              Retry
            </button>
          </div>
        )}

        <form className="fa-composer" onSubmit={onSubmit}>
          <textarea
            ref={inputRef}
            className="fa-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask about investing, markets or goals"
            aria-label="Message"
            rows={1}
            maxLength={2000}
          />
          {isStreaming ? (
            <button type="button" className="fa-send" onClick={stop} aria-label="Stop generating">
              <StopIcon />
            </button>
          ) : (
            <button type="submit" className="fa-send" disabled={!input.trim()} aria-label="Send message">
              <SendIcon />
            </button>
          )}
        </form>
        <p className="fa-disclaimer">{DISCLAIMER}</p>
      </div>
    </section>
  );
}
