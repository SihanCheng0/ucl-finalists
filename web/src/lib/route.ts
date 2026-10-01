// Selections live in the URL hash: #/explore?t=52280&s=2026, #/compare?a=52280:2026&b=52747:2026 (spec §6).

export type Screen = "pipeline" | "explore" | "compare" | "squad";
export interface Route { screen: Screen; params: Record<string, string> }
export const SCREENS: Screen[] = ["pipeline", "explore", "compare", "squad"];

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#\/?/, "");
  const cut = raw.indexOf("?");
  const path = cut === -1 ? raw : raw.slice(0, cut);
  const query = cut === -1 ? "" : raw.slice(cut + 1);
  const screen = (SCREENS as string[]).includes(path) ? (path as Screen) : "explore";
  const params: Record<string, string> = {};
  new URLSearchParams(query).forEach((value, key) => { params[key] = value; });
  return { screen, params };
}

export function formatHash(route: Route): string {
  const query = Object.entries(route.params)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value).replace(/%3A/gi, ":")}`)
    .join("&");
  return `#/${route.screen}${query ? `?${query}` : ""}`;
}

export interface Pick { teamId: string; season: number }

export function parsePick(text: string | undefined): Pick | null {
  if (!text) return null;
  const [teamId, season, extra] = text.split(":");
  const n = Number(season);
  return teamId && extra === undefined && Number.isInteger(n) && n > 2000 ? { teamId, season: n } : null;
}

export function formatPick(pick: Pick): string {
  return `${pick.teamId}:${pick.season}`;
}
