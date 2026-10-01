import { useEffect } from "react";
import { api } from "./api";
import { TopNav } from "./components/TopNav";
import { useApi } from "./hooks/useApi";
import { useHashRoute } from "./hooks/useHashRoute";
import { usePipeline } from "./hooks/usePipeline";
import { CompareScreen } from "./screens/CompareScreen";
import { ExploreScreen } from "./screens/ExploreScreen";
import { PipelineScreen } from "./screens/PipelineScreen";
import { SquadScreen } from "./screens/SquadScreen";

export function App() {
  const [route, navigate] = useHashRoute();
  const pipeline = usePipeline();
  const dataVersion = pipeline.view.finished; // every finished run may have rewritten the outputs
  const meta = useApi(() => api.meta(), [dataVersion]);
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
    content = <PipelineScreen meta={meta.data} pipeline={pipeline} />;
  } else if (!meta.data.ready) {
    content = <div className="empty"><h2>Run the pipeline first</h2>
      <p>{meta.data.problems.join(". ") || "Processed data or model outputs are missing."}</p>
      <a className="button primary" href="#/pipeline">Open the pipeline</a></div>;
  } else if (route.screen === "compare") {
    content = <CompareScreen meta={meta.data} route={route} navigate={navigate} dataVersion={dataVersion} />;
  } else if (route.screen === "squad") {
    content = <SquadScreen meta={meta.data} route={route} navigate={navigate} pipeline={pipeline} />;
  } else {
    content = <ExploreScreen meta={meta.data} route={route} navigate={navigate} dataVersion={dataVersion} />;
  }
  return (
    <div className="app">
      <TopNav screen={route.screen} view={pipeline.view} connected={pipeline.connected} />
      <main>{content}</main>
    </div>
  );
}
