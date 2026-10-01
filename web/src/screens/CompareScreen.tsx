import { api } from "../api";
import { SwapIcon } from "../components/Icons";
import { MirrorRow } from "../components/MirrorRow";
import { TeamPicker } from "../components/TeamPicker";
import { useApi } from "../hooks/useApi";
import { formatPick, parsePick, type Pick, type Route } from "../lib/route";
import type { Meta } from "../types";

export function CompareScreen({ meta, route, navigate, dataVersion }: {
  meta: Meta; route: Route; navigate: (route: Route) => void; dataVersion: number;
}) {
  const a = parsePick(route.params.a);
  const b = parsePick(route.params.b);
  const set = (key: "a" | "b", pick: Pick) => navigate({ screen: "compare", params: { ...route.params, [key]: formatPick(pick) } });
  const data = useApi(a && b ? () => api.compare(formatPick(a), formatPick(b)) : null, [route.params.a, route.params.b, dataVersion]);
  const features = new Map(meta.features.map((f) => [f.feature, f]));
  const rows = data.data?.rows ?? [];
  const tally = { a: rows.filter((r) => r.ahead === "a").length, b: rows.filter((r) => r.ahead === "b").length,
    tie: rows.filter((r) => r.ahead === "tie").length };
  return (
    <>
      <div className="pickers">
        <TeamPicker value={a} onChange={(pick) => set("a", pick)} side="a" label="First team" />
        <button className="icon-button" type="button" aria-label="Swap the teams" title="Swap the teams"
                disabled={!a && !b} onClick={() => navigate({ screen: "compare", params: { a: route.params.b ?? "", b: route.params.a ?? "" } })}>
          <SwapIcon />
        </button>
        <TeamPicker value={b} onChange={(pick) => set("b", pick)} side="b" label="Second team" />
      </div>
      {(!a || !b) && <div className="empty"><h2>Pick two team-seasons</h2>
        <p>Any club, any season since 2011-12, including the live one. Each stat is drawn against its own season's average,
          so teams from different seasons compare fairly.</p></div>}
      {data.error && <p className="error-note">{data.error.message}</p>}
      {data.data && (
        <section className="panel card" aria-label="Comparison">
          <div className="compare-head">
            <div><div className="picker-name" style={{ color: "var(--s1)" }}>{data.data.a.name}</div>
              <div className="small muted">{data.data.a.label}{data.data.a.live ? ", in progress" : ""}, {data.data.a.stage_label}</div></div>
            <div className="tally">{data.data.a.name} ahead on {tally.a}, {data.data.b.name} on {tally.b}, level on {tally.tie}</div>
            <div className="b"><div className="picker-name" style={{ color: "var(--s2)" }}>{data.data.b.name}</div>
              <div className="small muted">{data.data.b.label}{data.data.b.live ? ", in progress" : ""}, {data.data.b.stage_label}</div></div>
          </div>
          <div className="mirror">
            {meta.sections.map((section) => {
              const inSection = rows.filter((r) => r.section === section);
              if (!inSection.length) return null;
              return (
                <div key={section}>
                  <h3>{section}</h3>
                  {inSection.map((row) => {
                    const featureMeta = features.get(row.feature);
                    return featureMeta && <MirrorRow key={row.feature} row={row} meta={featureMeta}
                                                     nameA={data.data!.a.name} nameB={data.data!.b.name} />;
                  })}
                </div>
              );
            })}
          </div>
          <p className="small muted">Each bar is a team's distance from its own season's average, in standard deviations; the middle
            of each track is average and better always points outward. The team ahead on a stat is drawn solid.</p>
        </section>
      )}
    </>
  );
}
