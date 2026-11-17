import { useCallback, useRef } from "react";

const SESSION_KEY = "finance-assistant-session-id";

// crypto.randomUUID only exists in secure contexts (https or localhost),
// so fall back to getRandomValues when the app is opened over a LAN IP.
export function uuid(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

function loadOrCreate(): string {
  try {
    let id = localStorage.getItem(SESSION_KEY);
    if (!id) {
      id = uuid();
      localStorage.setItem(SESSION_KEY, id);
    }
    return id;
  } catch {
    return uuid(); // storage blocked: the session lasts until reload
  }
}

/**
 * The session ID is the LangGraph thread_id: the backend checkpointer keys
 * conversation memory by it. It is created once and kept in localStorage.
 */
export function useSession() {
  const idRef = useRef<string | null>(null);
  if (idRef.current === null) idRef.current = loadOrCreate();

  const getId = useCallback(() => idRef.current as string, []);

  /** Start a fresh conversation. Returns the previous ID. */
  const rotate = useCallback(() => {
    const previous = idRef.current as string;
    const next = uuid();
    idRef.current = next;
    try {
      localStorage.setItem(SESSION_KEY, next);
    } catch {
      /* storage blocked */
    }
    return previous;
  }, []);

  return { getId, rotate };
}
