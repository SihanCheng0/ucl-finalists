import { useEffect, useRef } from "react";
import { api } from "../api";
import type { CheckReport } from "../types";
import { useApi, type Loaded } from "./useApi";

const ON_FOCUS_AFTER_MS = 30_000;

/** The pre-run checks: run on load, after every finished pipeline run, and again when you come back to the tab. */
export function useChecks(dataVersion: number): Loaded<CheckReport> {
  const checks = useApi(() => api.checks(), [dataVersion]);
  const last = useRef(Date.now());
  const { reload, loading } = checks;
  useEffect(() => {
    if (!loading) last.current = Date.now();
  }, [loading]);
  useEffect(() => {
    const onFocus = () => {
      if (document.visibilityState === "visible" && Date.now() - last.current > ON_FOCUS_AFTER_MS) reload();
    };
    document.addEventListener("visibilitychange", onFocus);
    window.addEventListener("focus", onFocus);
    return () => { document.removeEventListener("visibilitychange", onFocus); window.removeEventListener("focus", onFocus); };
  }, [reload]);
  return checks;
}

/** What the run controls need to know about the AI model (LM Studio or OpenRouter). */
export function aiState(report: CheckReport | null): { ready: boolean; blocking: boolean; note: string } {
  if (!report) return { ready: true, blocking: false, note: "" };
  const ai = report.checks.filter((c) => c.group === "AI model");
  const failed = ai.find((c) => c.status === "fail");
  if (failed) return { ready: false, blocking: true, note: `${failed.label}: ${failed.detail}.` };
  const warned = ai.find((c) => c.status === "warn");
  if (warned) return { ready: false, blocking: false, note: `${warned.label}: ${warned.detail}.` };
  return { ready: true, blocking: false, note: "" };
}
