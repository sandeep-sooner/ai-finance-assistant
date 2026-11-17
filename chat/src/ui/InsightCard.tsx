import Markdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { InsightStatus } from "../hooks/useInsightStream";
import { DISCLAIMER } from "../lib/constants";

// react-markdown does not render raw HTML, so model output stays safe. Links open in a new tab.
const MD_COMPONENTS: Components = {
  a: ({ href, title, children }) => (
    <a href={href} title={title} target="_blank" rel="noreferrer">
      {children}
    </a>
  ),
};

interface Props {
  title: string;
  agentLabel: string;
  text: string;
  status: InsightStatus;
  error: string | null;
  onStop: () => void;
  onRegenerate: () => void;
}

/** The agent's streamed plain-language explanation, with stop / regenerate. */
export default function InsightCard({ title, agentLabel, text, status, error, onStop, onRegenerate }: Props) {
  const streaming = status === "streaming";
  return (
    <section className="ui-card ui-insight" aria-label={title}>
      <div className="ui-card-head">
        <h3>
          <span className="ui-agent-dot" aria-hidden="true" />
          {title}
          <span className="ui-muted ui-agent-name">{agentLabel}</span>
        </h3>
        {streaming ? (
          <button type="button" className="ui-btn" onClick={onStop}>
            Stop
          </button>
        ) : (
          status !== "idle" && (
            <button type="button" className="ui-btn" onClick={onRegenerate}>
              Regenerate
            </button>
          )
        )}
      </div>

      {streaming && !text ? (
        <div className="ui-dots" role="status" aria-label={`${agentLabel} is thinking`}>
          <span />
          <span />
          <span />
        </div>
      ) : (
        <div className="ui-md" aria-busy={streaming} aria-live="polite">
          <Markdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
            {text}
          </Markdown>
        </div>
      )}

      {status === "stopped" && <p className="ui-muted">Stopped.</p>}
      {error && (
        <p className="ui-error" role="alert">
          {error}
        </p>
      )}
      <p className="ui-disclaimer">{DISCLAIMER}</p>
    </section>
  );
}
