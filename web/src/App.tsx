import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { AboutDrawer } from "./components/AboutDrawer";
import { TopNav } from "./components/TopNav";
import { useApi } from "./hooks/useApi";
import { useChecks } from "./hooks/useChecks";
import { useHashRoute } from "./hooks/useHashRoute";
import { usePipeline } from "./hooks/usePipeline";
import { CompareScreen } from "./screens/CompareScreen";
import { ExploreScreen } from "./screens/ExploreScreen";
import { PipelineScreen } from "./screens/PipelineScreen";
import { PredictScreen } from "./screens/PredictScreen";
import { SquadScreen } from "./screens/SquadScreen";

export function App() {
  const [route, navigate] = useHashRoute();
  const pipeline = usePipeline();
  const dataVersion = pipeline.view.finished; // every finished run may have rewritten the outputs
  const meta = useApi(() => api.meta(), [dataVersion]);
  const checks = useChecks(dataVersion);
  const [about, setAbout] = useState(false);
  const openAbout = useCallback(() => setAbout(true), []);
  const closeAbout = useCallback(() => setAbout(false), []);
  const liveLoading = meta.data?.live_status === "loading";
  useEffect(() => {
    if (!liveLoading) return;
    const timer = window.setTimeout(meta.reload, 4000); // the live season loads in the background at startup
    return () => window.clearTimeout(timer);
  }, [liveLoading, meta.reload, meta.data]);

  let content;
  if (meta.error) {
    content = <div className="empty"><h2>Can't reach the UCL Lab server</h2><p>{meta.error.message}</p>
      <button className="button" type="button" onClick={meta.reload}>Try again</button></div>;
  } else if (!meta.data) {
    content = <div className="empty"><p>Loading…</p></div>;
  } else if (route.screen === "pipeline") {
    content = <PipelineScreen meta={meta.data} pipeline={pipeline} checks={checks} />;
  } else if (!meta.data.ready) {
    content = <div className="empty"><h2>Run the pipeline first</h2>
      <p>{meta.data.problems.join(". ") || "Processed data or model outputs are missing."}</p>
      <a className="button primary" href="#/pipeline">Open the pipeline</a></div>;
  } else if (route.screen === "compare") {
    content = <CompareScreen meta={meta.data} route={route} navigate={navigate} dataVersion={dataVersion} />;
  } else if (route.screen === "predict") {
    content = <PredictScreen meta={meta.data} route={route} navigate={navigate} dataVersion={dataVersion} />;
  } else if (route.screen === "squad") {
    content = <SquadScreen meta={meta.data} route={route} navigate={navigate} pipeline={pipeline} />;
  } else {
    content = <ExploreScreen meta={meta.data} route={route} navigate={navigate} dataVersion={dataVersion} onAbout={openAbout} />;
  }
  return (
    <div className="app">
      <TopNav screen={route.screen} view={pipeline.view} connected={pipeline.connected} checks={checks.data}
              onAbout={openAbout} />
      <main>{content}</main>
      {about && meta.data && <AboutDrawer meta={meta.data} onClose={closeAbout} />}
    </div>
  );
}
