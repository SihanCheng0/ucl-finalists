import { useEffect, useState } from "react";
import type { Screen } from "../lib/route";
import { ago } from "../lib/format";
import type { PipelineView } from "../lib/pipelineState";
import { STATIC_SITE } from "../site";
import type { CheckReport } from "../types";
import { AutoIcon, InfoIcon, Mark, MoonIcon, SunIcon } from "./Icons";

const TABS: { screen: Screen; label: string }[] = [
  { screen: "pipeline", label: "Pipeline" },
  { screen: "explore", label: "Explore" },
  { screen: "compare", label: "Compare" },
  { screen: "squad", label: "Squad" },
  { screen: "predict", label: "Predict" },
];
type Theme = "system" | "light" | "dark";
const NEXT: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };

function storedTheme(): Theme {
  try {
    const value = localStorage.getItem("ucl-lab-theme");
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

export function runLabel(view: PipelineView, connected: boolean, publishedAt: string | null = null): { tone: string; text: string } {
  if (STATIC_SITE) {
    if (view.outcome?.status === "failed") return { tone: "failed", text: "Last nightly run failed" };
    if (view.outcome?.status === "warning") return { tone: "warning", text: `Updated ${ago(publishedAt)}, with warnings` };
    return { tone: "done", text: publishedAt ? `Updated ${ago(publishedAt)}` : "Loading…" };
  }
  if (!connected && view.state === null) return { tone: "offline", text: "Server offline" };
  const state = view.state;
  if (state?.running) {
    const stage = state.stages.find((s) => s.status === "running");
    return { tone: "running", text: stage ? `Running ${stage.name}` : "Running" };
  }
  if (view.outcome?.status === "failed") return { tone: "failed", text: "Last run failed" };
  if (view.outcome?.status === "warning") return { tone: "warning", text: "Last run had warnings" };
  return { tone: "done", text: "Idle" };
}

function healthLabel(checks: CheckReport | null): { tone: string; text: string } | null {
  if (!checks) return null;
  const { warn, fail } = checks.summary;
  if (fail) return { tone: "failed", text: `${fail} ${fail === 1 ? "problem" : "problems"} to fix` };
  if (warn) return { tone: "warning", text: `${warn} ${warn === 1 ? "check" : "checks"} to look at` };
  return { tone: "done", text: "All checks passed" };
}

export function TopNav({ screen, view, connected, publishedAt, checks, onAbout }: {
  screen: Screen; view: PipelineView; connected: boolean; publishedAt: string | null; checks: CheckReport | null;
  onAbout: () => void;
}) {
  const [theme, setTheme] = useState<Theme>(storedTheme);
  useEffect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    try {
      if (theme === "system") localStorage.removeItem("ucl-lab-theme");
      else localStorage.setItem("ucl-lab-theme", theme);
    } catch {
      /* the choice just isn't remembered */
    }
  }, [theme]);
  const run = runLabel(view, connected, publishedAt);
  const health = healthLabel(checks);
  const ThemeIcon = theme === "light" ? SunIcon : theme === "dark" ? MoonIcon : AutoIcon;
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <a className="wordmark" href="#/explore" aria-label="UCL Lab"><Mark /><span className="name">UCL Lab</span></a>
        <nav className="nav" aria-label="Screens">
          {TABS.map((tab) => (
            <a key={tab.screen} href={`#/${tab.screen}`} aria-current={tab.screen === screen ? "page" : undefined}>{tab.label}</a>
          ))}
        </nav>
        <div className="topbar-end">
          {health && (
            <a className="run-pill" href="#/pipeline" title="Checks of the AI model, UEFA and the outputs">
              <span className={`dot ${health.tone}`} /><span className="text">{health.text}</span>
            </a>
          )}
          <a className="run-pill" href="#/pipeline" title={STATIC_SITE ? "The nightly run that published this site" : "Pipeline status"}>
            <span className={`dot ${run.tone}`} /><span className="text">{run.text}</span>
          </a>
          <button className="about-button" type="button" onClick={onAbout} aria-haspopup="dialog">
            <InfoIcon /><span className="text">About</span>
          </button>
          <button className="icon-button" type="button" onClick={() => setTheme(NEXT[theme])}
                  aria-label={`Theme: ${theme}. Switch to ${NEXT[theme]}`} title={`Theme: ${theme}`}>
            <ThemeIcon />
          </button>
        </div>
      </div>
    </header>
  );
}
