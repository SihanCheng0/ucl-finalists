import { Fragment, useState } from "react";
import { api, ApiError } from "../api";
import type { Loaded } from "../hooks/useApi";
import { ago } from "../lib/format";
import { STATIC_SITE } from "../site";
import type { CheckItem, CheckReport, ModelTest } from "../types";
import { AlertIcon, CheckIcon, CrossIcon } from "./Icons";

const GROUP_ORDER = ["AI model", "UEFA feeds", "Pipeline outputs", "Live season and players", "Dashboard"];

function StatusMark({ status }: { status: CheckItem["status"] }) {
  const Icon = status === "ok" ? CheckIcon : status === "fail" ? CrossIcon : status === "warn" ? AlertIcon : null;
  return <span className={`check-mark ${status}`} aria-label={status === "info" ? "note" : status}>{Icon ? <Icon /> : "i"}</span>;
}

/** Text with `commands` in backticks shown as code, so a fix can be copied as it stands. */
function WithCode({ text }: { text: string }) {
  return <>{text.split(/(`[^`]+`)/).map((part, i) => part.startsWith("`") && part.endsWith("`")
    ? <code key={i}>{part.slice(1, -1)}</code> : <Fragment key={i}>{part}</Fragment>)}</>;
}

export function summaryText(report: CheckReport): string {
  const { ok, warn, fail } = report.summary;
  const parts = [`${ok} passed`];
  if (warn) parts.push(`${warn} to look at`);
  if (fail) parts.push(`${fail} ${fail === 1 ? "problem" : "problems"}`);
  return parts.join(", ");
}

/** "Before you run": everything a run depends on, with what to do about anything that isn't ready. On the website,
 * the checks the nightly run made before publishing it. */
export function Checks({ checks }: { checks: Loaded<CheckReport> }) {
  const report = checks.data;
  const trouble = !!report && (report.summary.warn > 0 || report.summary.fail > 0);
  const [open, setOpen] = useState<boolean | null>(null);
  const [test, setTest] = useState<{ busy: boolean; result: ModelTest | null; error: string | null }>(
    { busy: false, result: null, error: null });
  const expanded = open ?? trouble;
  const runTest = async () => {
    setTest({ busy: true, result: null, error: null });
    try {
      const result = await api.testModel(true);
      setTest({ busy: false, result, error: null });
      checks.reload();
    } catch (error) {
      setTest({ busy: false, result: null, error: error instanceof ApiError ? error.message : String(error) });
    }
  };
  const groups = GROUP_ORDER.map((group) => ({ group, items: (report?.checks ?? []).filter((c) => c.group === group) }))
    .filter((g) => g.items.length);
  return (
    <section className="panel checks" aria-label={STATIC_SITE ? "Checks" : "Before you run"}>
      <div className="checks-head">
        <div>
          <h3>{STATIC_SITE ? "Checks from the nightly run" : "Before you run"}</h3>
          <p className="small muted" aria-live="polite">
            {checks.loading && !report ? "Checking the AI model, UEFA and the outputs…"
              : report ? <>{summaryText(report)}{checks.loading ? ", checking again…" : `, checked ${ago(report.checked_at)}`}</>
                : checks.error ? checks.error.message : ""}
          </p>
        </div>
        <div className="checks-actions">
          {!STATIC_SITE && <>
            <button className="button small" type="button" onClick={checks.reload} disabled={checks.loading}>Check again</button>
            <button className="button small" type="button" onClick={runTest} disabled={test.busy}>
              {test.busy ? "Testing the model…" : report?.provider === "openrouter" ? "Test the hosted model" : "Test the local model"}</button>
          </>}
          {report && <button className="button small" type="button" aria-expanded={expanded} onClick={() => setOpen(!expanded)}>
            {expanded ? "Hide details" : `Show all ${report.checks.length} checks`}</button>}
        </div>
      </div>
      {(test.busy || test.result || test.error) && (
        <div className={`model-test ${test.result?.ok ? "ok" : test.busy ? "busy" : "fail"}`} role="status">
          {test.busy && (report?.provider === "openrouter"
            ? <>Checking the OpenRouter key and asking {report.model} one question.</>
            : <>Loading {report?.model ?? "the model"} if needed and asking it one question. The first load can take a minute or two.</>)}
          {test.result && (test.result.ok
            ? <>{test.result.detail}{test.result.reply ? <>, replying “{test.result.reply}”</> : null}
                {test.result.load_seconds && test.result.load_seconds > 1 ? ` after ${test.result.load_seconds} s getting ready` : ""}.</>
            : <>{test.result.detail}</>)}
          {test.error && <>{test.error}</>}
        </div>
      )}
      {expanded && report && (
        <div className="check-groups">
          {groups.map(({ group, items }) => (
            <div key={group} className="check-group">
              <h4>{group}</h4>
              {items.map((item) => (
                <div key={item.id} className={`check-row ${item.status}`}>
                  <StatusMark status={item.status} />
                  <div className="check-text">
                    <div><span className="check-label">{item.label}</span> <span className="muted">{item.detail}</span></div>
                    {item.fix && item.status !== "ok" && <div className="check-fix"><WithCode text={item.fix} /></div>}
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
