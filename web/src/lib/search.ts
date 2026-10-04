// Team search in the browser, for the website: the same matching and order as queries.search on the server,
// over the index that queries.search_index writes to data/teams.json.
import type { TeamHit } from "../types";

export interface TeamEntry extends TeamHit { names: string[]; key: string; latest: number }

export const SEARCH_LIMIT = 12;
const MIN_QUERY = 2;
// queries.LETTERS, plus ß, which Python's casefold() turns into ss and toLowerCase() leaves alone
const LETTERS: Record<string, string> = { "ø": "o", "æ": "ae", "œ": "oe", "ł": "l", "đ": "d", "ı": "i", "þ": "th",
  "ð": "d", "ß": "ss" };

/** queries.fold: lower case without accents or punctuation, so "Atlético" matches "atletico". */
export function fold(text: string): string {
  const lowered = [...text.toLowerCase()].map((c) => LETTERS[c] ?? c).join("");
  const plain = lowered.normalize("NFKD").replace(/\p{M}/gu, "");
  return plain.replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

/** Names that start with the query first, then the most recent season, then the name. */
export function searchTeams(index: TeamEntry[], q: string, limit = SEARCH_LIMIT): TeamHit[] {
  const query = fold(q);
  if (query.length < MIN_QUERY) return [];
  const hits: { rank: number; team: TeamEntry }[] = [];
  for (const team of index) {
    if (team.names.some((name) => name.startsWith(query))) hits.push({ rank: 0, team });
    else if (team.names.some((name) => name.includes(query))) hits.push({ rank: 1, team });
  }
  hits.sort((x, y) => x.rank - y.rank || y.team.latest - x.team.latest
    || (x.team.key < y.team.key ? -1 : x.team.key > y.team.key ? 1 : 0));
  return hits.slice(0, limit).map(({ team }) => ({ team_id: team.team_id, name: team.name, seasons: team.seasons }));
}
