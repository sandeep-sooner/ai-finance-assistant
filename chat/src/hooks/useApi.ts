import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

interface State<T> {
  data: T | undefined;
  error: string | null;
  loading: boolean;
  /** The key the current data was loaded for (data is kept while a new key loads). */
  dataKey: string | null;
}

interface Options {
  /** Refetch on this interval while refreshEnabled is true. */
  refreshMs?: number;
  /** Pause polling, e.g. while the tab is hidden. */
  refreshEnabled?: boolean;
}

function message(err: unknown): string {
  return err instanceof ApiError ? err.message : "Something went wrong.";
}

/**
 * Fetch whenever `key` changes (null = do nothing). Keeps previous data while a
 * new request is in flight, aborts superseded requests, and can poll.
 */
export function useApi<T>(key: string | null, fetcher: (signal: AbortSignal) => Promise<T>, options: Options = {}) {
  const { refreshMs, refreshEnabled = true } = options;
  const [state, setState] = useState<State<T>>({ data: undefined, error: null, loading: key !== null, dataKey: null });
  const [nonce, setNonce] = useState(0);
  const fetcherRef = useRef(fetcher);

  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  useEffect(() => {
    if (key === null) return;
    const controller = new AbortController();
    const run = async () => {
      setState((s) => ({ ...s, loading: true, error: null }));
      try {
        const data = await fetcherRef.current(controller.signal);
        if (!controller.signal.aborted) setState({ data, error: null, loading: false, dataKey: key });
      } catch (err) {
        if (controller.signal.aborted || (err as Error).name === "AbortError") return;
        setState((s) => ({ ...s, error: message(err), loading: false }));
      }
    };
    void run();
    return () => controller.abort();
  }, [key, nonce]);

  useEffect(() => {
    if (key === null || !refreshMs || !refreshEnabled) return;
    const id = window.setInterval(() => setNonce((n) => n + 1), refreshMs);
    return () => window.clearInterval(id);
  }, [key, refreshMs, refreshEnabled]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, reload };
}
