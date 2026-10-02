import { api } from "../api";
import { useApi } from "../hooks/useApi";
import { num } from "../lib/format";
import { chance, titleLine, versions } from "../lib/predict";
import { formatPick, type Pick } from "../lib/route";
import type { Venue } from "../types";

const VENUES: { venue: Venue; label: (a: string, b: string) => string }[] = [
  { venue: "neutral", label: () => "Neutral venue" },
  { venue: "a", label: (a) => `At ${a}` },
  { venue: "b", label: (_, b) => `At ${b}` },
];

/** One prediction for two team-seasons: a match between them, a two-legged tie, a final, and which club is more
 * likely to win this season's Champions League. */
export function HeadToHeadCard({ a, b, venue, onVenue, dataVersion }: {
  a: Pick; b: Pick; venue: Venue; onVenue: (venue: Venue) => void; dataVersion: number;
}) {
  const h2h = useApi(() => api.headToHead(formatPick(a), formatPick(b), venue),
                     [formatPick(a), formatPick(b), venue, dataVersion]);
  const d = h2h.data;
  const title = d && titleLine(d.a, d.b, d.title_label);
  const who = d && versions(d.a, d.b);
  const likeliest = d?.likely_scores[0];
  return (
    <section className="panel card h2h" aria-label="Head to head prediction">
      <div className="card-title">
        <h3>If they met</h3>
        {d && (
          <div className="segmented" role="group" aria-label="Where the match is played">
            {VENUES.map((v) => (
              <button key={v.venue} type="button" aria-pressed={venue === v.venue} onClick={() => onVenue(v.venue)}>
                {v.label(d.a.name, d.b.name)}
              </button>
            ))}
          </div>
        )}
      </div>
      {h2h.error && <p className="error-note">{h2h.error.message}</p>}
      {!d && !h2h.error && <p className="muted small">Working out the odds…</p>}
      {d && (
        <>
          <div className="h2h-split" aria-hidden>
            <span style={{ color: "var(--s1)" }}>{chance(d.win)}</span>
            <span className="draw">{chance(d.draw)} draw</span>
            <span className="b" style={{ color: "var(--s2)" }}>{chance(d.loss)}</span>
          </div>
          <div className="h2h-bar" role="img"
               aria-label={`${d.a.name} win ${chance(d.win)}, draw ${chance(d.draw)}, ${d.b.name} win ${chance(d.loss)}`}>
            <span style={{ flexGrow: d.win, background: "var(--s1)" }} />
            <span style={{ flexGrow: d.draw, background: "var(--rule)" }} />
            <span style={{ flexGrow: d.loss, background: "var(--s2)" }} />
          </div>
          <div className="h2h-names small">
            <span>{d.a.name}</span><span className="draw">Draw</span><span className="b">{d.b.name}</span>
          </div>
          <div className="figures">
            <div className="figure"><div className="label">Likeliest score</div>
              <div className="value">{likeliest ? `${likeliest.a}–${likeliest.b}` : "–"}</div>
              <div className="small muted">{likeliest ? `${chance(likeliest.p)} chance` : ""}</div></div>
            <div className="figure"><div className="label">Expected goals</div>
              <div className="value">{num(d.xg_a, 1)}–{num(d.xg_b, 1)}</div>
              <div className="small muted">{d.a.name} first</div></div>
            <div className="figure"><div className="label">Over two legs</div>
              <div className="value">{chance(d.tie)}</div>
              <div className="small muted">{d.a.name} to go through</div></div>
            <div className="figure"><div className="label">In a final</div>
              <div className="value">{chance(d.final)}</div>
              <div className="small muted">{d.a.name} to win it</div></div>
          </div>
          {title && <p className="h2h-title">{title}</p>}
          <p className="small muted">
            {who ? `${who} ` : ""}Ratings {d.a.rating.toLocaleString("en-US")} and {d.b.rating.toLocaleString("en-US")}, from
            every Champions League match since 2011-12. A two-legged tie and a final count rating gaps for less, as
            knockout football does: <a href="#/predict">how good these odds are</a>.
          </p>
        </>
      )}
    </section>
  );
}
