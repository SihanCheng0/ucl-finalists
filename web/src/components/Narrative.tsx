import type { NarrativeData } from "../types";

const ICON: Record<string, string> = { good: "✓", warn: "!", stale: "↻", unavailable: "–" };

export function Badge({ badge }: { badge: NarrativeData["badge"] }) {
  return <span className={`badge ${badge.kind}`}><span aria-hidden>{ICON[badge.kind]}</span>{badge.label}</span>;
}

/** The AI model's write-up. The server escapes the model's text before it becomes HTML (report.md_to_html). */
export function Narrative({ narrative, title = "AI scouting report" }: { narrative: NarrativeData; title?: string }) {
  return (
    <section className="panel card" aria-label={title}>
      <div className="card-title"><h3>{title}</h3><Badge badge={narrative.badge} /></div>
      {narrative.html
        ? <div className="narrative" dangerouslySetInnerHTML={{ __html: narrative.html }} />
        : <p className="muted small">The AI model couldn't write this one. Run the analyze stage again once the AI model checks pass.</p>}
    </section>
  );
}
