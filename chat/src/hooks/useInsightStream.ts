import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, openStream } from "../api/client";
import { parseSSE } from "./useChatStream";

export type InsightStatus = "idle" | "streaming" | "done" | "stopped" | "error";

function parseData(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    return data;
  }
}

/**
 * Streams an agent's explanation from an SSE endpoint that emits
 * agent -> token* -> sources -> done (or error). Used by the Portfolio and Goals tabs.
 */
export function useInsightStream<P>(path: string) {
  const [text, setText] = useState("");
  const [status, setStatus] = useState<InsightStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const lastPayload = useRef<P | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const start = useCallback(
    async (payload: P) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      lastPayload.current = payload;
      setText("");
      setError(null);
      setStatus("streaming");

      try {
        const res = await openStream(path, payload, controller.signal);
        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finished = false;
        try {
          while (!finished) {
            const { done, value } = await reader.read();
            buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
            const parsed = parseSSE(done ? buffer + "\n\n" : buffer);
            buffer = parsed.rest;
            for (const e of parsed.events) {
              const v = parseData(e.data);
              if (e.event === "token" && typeof v === "string") setText((t) => t + v);
              else if (e.event === "error") throw new ApiError(typeof v === "string" ? v : "The explanation failed.", 500);
              else if (e.event === "done") finished = true;
            }
            if (done) break;
          }
        } finally {
          reader.cancel().catch(() => {});
        }
        if (!finished) throw new ApiError("The connection closed before the explanation finished.", 0);
        if (abortRef.current === controller) setStatus("done");
      } catch (err) {
        if (abortRef.current !== controller) return; // superseded or cleared
        if ((err as Error).name === "AbortError") {
          setStatus("stopped");
        } else {
          setError(err instanceof ApiError ? err.message : "Something went wrong.");
          setStatus("error");
        }
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [path]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const regenerate = useCallback(() => {
    if (lastPayload.current !== null) void start(lastPayload.current);
  }, [start]);

  const clear = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setText("");
    setError(null);
    setStatus("idle");
  }, []);

  return { text, status, error, start, stop, regenerate, clear };
}
