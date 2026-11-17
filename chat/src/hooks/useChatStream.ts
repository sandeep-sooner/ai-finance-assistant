import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, deleteSession, openChatStream } from "../api/client";
import type { ChatMessage } from "../chat/types";
import type { ApiErrorBody, Level, Source, StreamDonePayload, StreamEventName, StreamMeta } from "../types/api";
import { useSession, uuid } from "./useSession";

// Abort if the server sends nothing for this long (router + RAG + LLM can be slow).
const IDLE_TIMEOUT_MS = 90_000;

interface SSEEvent {
  event: string;
  data: string;
}

// Split a text buffer into complete SSE events plus the unfinished remainder.
export function parseSSE(buffer: string): { events: SSEEvent[]; rest: string } {
  const blocks = buffer.replace(/\r\n?/g, "\n").split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: SSEEvent[] = [];
  for (const block of blocks) {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue; // comment / keep-alive
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1); // the spec strips exactly one space
      if (field === "event") event = value.trim();
      else if (field === "data") dataLines.push(value);
    }
    if (dataLines.length || event !== "message") events.push({ event, data: dataLines.join("\n") });
  }
  return { events, rest };
}

// Data is JSON per the contract; fall back to the raw text if a server sends plain strings.
function parseData(data: string): unknown {
  if (data === "") return "";
  try {
    return JSON.parse(data);
  } catch {
    return data;
  }
}

function toAgents(value: unknown): string[] | undefined {
  if (Array.isArray(value)) return value.map(String).filter(Boolean);
  if (typeof value === "string" && value) return value.split(",").map((s) => s.trim()).filter(Boolean);
  return undefined;
}

function errorMessage(value: unknown): string {
  if (typeof value === "string" && value) return value;
  if (value && typeof value === "object" && "message" in value) return String((value as ApiErrorBody).message);
  return "The assistant ran into a problem.";
}

class StreamError extends Error {}

export function useChatStream(level: Level = "beginner") {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const levelRef = useRef(level); // read at send time, so changing it never re-creates `send`
  useEffect(() => {
    levelRef.current = level;
  }, [level]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const session = useSession();
  const abortRef = useRef<AbortController | null>(null);
  const busyRef = useRef(false); // guards against double sends before state updates
  const lastUserText = useRef("");

  // Cancel any in-flight request when the component unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  const patchMessage = useCallback((id: string, patch: (m: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? patch(m) : m)));
  }, []);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || busyRef.current) return;
      busyRef.current = true;

      lastUserText.current = trimmed;
      setError(null);

      const assistantId = uuid();
      setMessages((prev) => [
        ...prev,
        { id: uuid(), role: "user", content: trimmed },
        { id: assistantId, role: "assistant", content: "" },
      ]);

      const controller = new AbortController();
      abortRef.current = controller;
      setIsStreaming(true);

      let finished = false;
      let timedOut = false;
      let idleTimer: number | undefined;
      const resetIdle = () => {
        window.clearTimeout(idleTimer);
        idleTimer = window.setTimeout(() => {
          timedOut = true;
          controller.abort();
        }, IDLE_TIMEOUT_MS);
      };

      const handleEvent = ({ event, data }: SSEEvent) => {
        const value = parseData(data);
        switch (event as StreamEventName) {
          case "token":
            if (typeof value === "string" && value) {
              patchMessage(assistantId, (m) => ({ ...m, content: m.content + value }));
            }
            break;
          case "agent": {
            const agents = toAgents(value);
            if (agents) patchMessage(assistantId, (m) => ({ ...m, agents }));
            break;
          }
          case "sources":
            if (Array.isArray(value)) patchMessage(assistantId, (m) => ({ ...m, sources: value as Source[] }));
            break;
          case "meta":
            if (value && typeof value === "object") {
              const meta = value as StreamMeta;
              patchMessage(assistantId, (m) => ({ ...m, meta, ...(meta.disclaimer && { disclaimer: meta.disclaimer }) }));
            }
            break;
          case "done": {
            finished = true;
            const p = (value && typeof value === "object" ? value : {}) as StreamDonePayload;
            const agents = toAgents(p.agent_used);
            patchMessage(assistantId, (m) => ({
              ...m,
              ...(agents && { agents }),
              ...(Array.isArray(p.sources) && { sources: p.sources }),
              ...(p.disclaimer && { disclaimer: p.disclaimer }),
            }));
            break;
          }
          case "error":
            throw new StreamError(errorMessage(value));
        }
      };

      try {
        resetIdle();
        const res = await openChatStream({ session_id: session.getId(), message: trimmed, level: levelRef.current }, controller.signal);
        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        try {
          while (!finished) {
            const { done, value } = await reader.read();
            resetIdle();
            buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
            // At end of stream, also flush a final event that lacks its trailing blank line.
            const parsed = parseSSE(done ? buffer + "\n\n" : buffer);
            buffer = parsed.rest;
            for (const e of parsed.events) handleEvent(e);
            if (done) break;
          }
        } finally {
          reader.cancel().catch(() => {});
        }

        if (!finished) throw new StreamError("The connection closed before the answer finished.");
      } catch (err) {
        if ((err as Error).name === "AbortError" && !timedOut) {
          // User pressed Stop or started a new chat: keep whatever text arrived.
        } else {
          const message = timedOut
            ? "The assistant took too long to respond."
            : err instanceof ApiError || err instanceof StreamError
              ? err.message
              : "Something went wrong.";
          setError(message);
          patchMessage(assistantId, (m) => ({ ...m, failed: true }));
        }
      } finally {
        window.clearTimeout(idleTimer);
        if (abortRef.current === controller) abortRef.current = null;
        busyRef.current = false;
        setIsStreaming(false);
      }
    },
    [patchMessage, session]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const retry = useCallback(() => {
    if (busyRef.current) return;
    // Drop the failed assistant reply and the user message that caused it, then resend.
    setMessages((prev) => prev.slice(0, -2));
    void send(lastUserText.current);
  }, [send]);

  // New chat: start a new server session so the old history is not carried over.
  const reset = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setError(null);
    const previous = session.rotate();
    deleteSession(previous).catch(() => {
      /* best effort: the backend may not expose this endpoint yet */
    });
  }, [session]);

  return { messages, isStreaming, error, send, stop, retry, reset };
}
