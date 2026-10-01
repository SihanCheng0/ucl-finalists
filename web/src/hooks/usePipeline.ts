import { useCallback, useEffect, useState } from "react";
import { api, ApiError, EVENTS_URL } from "../api";
import { applyEvent, initialView, withState, type PipelineView } from "../lib/pipelineState";
import type { PipelineEvent } from "../types";

const TYPES = ["stage", "progress", "log", "done", "run_failed"] as const;
const POLL_MS = 3000; // while a run is going, /state catches anything the coalesced progress events held back

export interface Pipeline {
  view: PipelineView;
  connected: boolean;
  start: (body: { stages?: string[]; skip_ai?: boolean; refresh_live?: boolean }) => Promise<boolean>;
  startError: string | null;
}

/** /state first, then the event stream (its backlog fills the log). EventSource reconnects by itself and sends the
 * last id it saw; a new boot means the server restarted, and the reducer then starts over. */
export function usePipeline(): Pipeline {
  const [view, setView] = useState<PipelineView>(initialView);
  const [connected, setConnected] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  useEffect(() => {
    let source: EventSource | null = null;
    let closed = false;
    let retry: number | undefined;
    const connect = () => {
      api.pipelineState().then((state) => {
        if (closed) return;
        setView((v) => withState(v, state));
        source = new EventSource(EVENTS_URL);
        source.onopen = () => setConnected(true);
        source.onerror = () => setConnected(false);
        for (const type of TYPES) {
          source.addEventListener(type, (message) => {
            const event = JSON.parse((message as MessageEvent<string>).data) as PipelineEvent;
            setView((v) => applyEvent(v, event));
          });
        }
      }, () => {
        setConnected(false);
        if (!closed) retry = window.setTimeout(connect, 2000);
      });
    };
    connect();
    return () => { closed = true; window.clearTimeout(retry); source?.close(); };
  }, []);

  const needsState = view.boot !== null && view.state === null;
  const running = view.state?.running ?? false;
  useEffect(() => {
    if (!needsState && !running) return;
    const refresh = () => api.pipelineState().then((state) => setView((v) => withState(v, state)), () => undefined);
    if (needsState) refresh();
    const timer = running ? window.setInterval(refresh, POLL_MS) : undefined;
    return () => window.clearInterval(timer);
  }, [needsState, running]);

  const start = useCallback(async (body: { stages?: string[]; skip_ai?: boolean; refresh_live?: boolean }) => {
    setStartError(null);
    try {
      await api.startRun(body);
      return true;
    } catch (error) {
      setStartError(error instanceof ApiError ? error.message : String(error));
      return false;
    }
  }, []);

  return { view, connected, start, startError };
}
