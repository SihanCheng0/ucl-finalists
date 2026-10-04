// The website build (`npm run build:site`) reads static files that `ucl publish` writes, and runs nothing: the
// pipeline runs once a day on GitHub instead. The local build talks to `uv run ucl web`.
export const STATIC_SITE: boolean = import.meta.env.VITE_STATIC === "1";
export const REPO_URL = "https://github.com/SihanCheng0/ucl-finalists";
export const NIGHTLY_URL = `${REPO_URL}/actions/workflows/nightly.yml`;
