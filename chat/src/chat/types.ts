import type { Source, StreamMeta } from "../types/api";

export type { Source, StreamEventName } from "../types/api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** One agent normally; several when the router fans out a multi-agent query. */
  agents?: string[];
  sources?: Source[];
  disclaimer?: string;
  /** How the answer was produced: route, model tier, cache hit, fallbacks. */
  meta?: StreamMeta;
  failed?: boolean;
}
