// Head-to-heads in the browser, for the website: forecast.head_to_head ported line for line, fed by the ratings and
// goals models that ForecastService.matchups writes to data/matchups.json. h2h.fixture.json holds cases worked out by
// the Python, and h2h.test.ts holds this file to them (tests/test_h2h_fixture.py keeps the fixture current).
import type { HeadToHead, HeadToHeadSide, Venue } from "../types";

export interface Goals { base: number; slope: number }
export interface Models { league: Goals; early: Goals; late: Goals }
export interface Matchups {
  home: number; max_goals: number; models: Models; title_label: string | null;
  sides: Record<string, HeadToHeadSide & { exact: number }>;
}
export type Outcome = Omit<HeadToHead, "a" | "b" | "title_label">;

type Matrix = number[][];

/** Goals.rates: log(expected goals) = base ± slope × rating gap / 400, for the home and the away side. */
function rates(goals: Goals, diff: number): [number, number] {
  const x = diff / 400;
  return [Math.exp(goals.base + goals.slope * x), Math.exp(goals.base - goals.slope * x)];
}

function logFactorials(n: number): number[] {
  const out = [0];
  for (let k = 1; k <= n; k++) out.push(out[k - 1] + Math.log(k));
  return out;
}

function poisson(rate: number, maxGoals: number): number[] {
  return logFactorials(maxGoals).map((logFact, k) => Math.exp(-rate + k * Math.log(rate) - logFact));
}

/** P(home scores i, away scores j) for i, j up to maxGoals, renormalised. */
export function scoreMatrix(homeRate: number, awayRate: number, maxGoals: number): Matrix {
  const ph = poisson(homeRate, maxGoals), pa = poisson(awayRate, maxGoals);
  const m = ph.map((p) => pa.map((q) => p * q));
  const total = m.reduce((sum, row) => sum + row.reduce((s, v) => s + v, 0), 0);
  return m.map((row) => row.map((v) => v / total));
}

/** (home win, draw, away win). */
export function outcome(m: Matrix): [number, number, number] {
  let win = 0, draw = 0, loss = 0;
  m.forEach((row, i) => row.forEach((v, j) => {
    if (i > j) win += v;
    else if (i === j) draw += v;
    else loss += v;
  }));
  return [win, draw, loss];
}

/** The first side's goals minus the second's, indexed from -maxGoals. */
function margin(m: Matrix, maxGoals: number): number[] {
  const out = new Array(2 * maxGoals + 1).fill(0);
  m.forEach((row, i) => row.forEach((v, j) => { out[i - j + maxGoals] += v; }));
  return out;
}

function convolve(a: number[], b: number[]): number[] {
  const out = new Array(a.length + b.length - 1).fill(0);
  a.forEach((x, i) => b.forEach((y, j) => { out[i + j] += x * y; }));
  return out;
}

/** Chance the first side wins after a level 90 (or 180) minutes: 30 more minutes, then a penalty coin flip. */
function extraTimeWin(diff: number, goals: Goals, maxGoals: number): number {
  const [rh, ra] = rates(goals, diff);
  const [win, draw] = outcome(scoreMatrix(rh / 3, ra / 3, maxGoals));
  return win + draw / 2;
}

/** Chance A goes through a two-legged tie: 180 minutes, then extra time at the second leg's ground, then penalties. */
export function tieProbability(gap: number, goals: Goals, home: number, aHostsSecond: boolean, maxGoals: number): number {
  const first = scoreMatrix(...rates(goals, gap + (aHostsSecond ? -home : home)), maxGoals);
  const second = scoreMatrix(...rates(goals, gap + (aHostsSecond ? home : -home)), maxGoals);
  const total = convolve(margin(first, maxGoals), margin(second, maxGoals)); // indexed from -2 * maxGoals
  const ahead = total.slice(2 * maxGoals + 1).reduce((s, v) => s + v, 0);
  const level = total[2 * maxGoals];
  return ahead + level * extraTimeWin(gap + (aHostsSecond ? home : -home), goals, maxGoals);
}

/** A against B three ways: one match like a league-phase game, a two-legged tie, and a final. */
export function headToHead(ratingA: number, ratingB: number, models: Models, home: number, venue: Venue,
                           maxGoals: number): Outcome {
  const gap = ratingA - ratingB;
  const diff = gap + { neutral: 0, a: home, b: -home }[venue];
  const [xgA, xgB] = rates(models.league, diff);
  const m = scoreMatrix(xgA, xgB, maxGoals);
  const [win, draw, loss] = outcome(m);
  // likeliest first; equally likely scores in the grid's reading order (0-1 before 1-0), as forecast.py sorts them
  const cells = m.flatMap((row, i) => row.map((p, j) => ({ a: i, b: j, p })));
  const likely = cells.map((cell, index) => ({ cell, index }))
    .sort((x, y) => y.cell.p - x.cell.p || x.index - y.index).slice(0, 3).map(({ cell }) => cell);
  const [finalWin, finalDraw] = outcome(scoreMatrix(...rates(models.late, gap), maxGoals));
  return {
    venue, win, draw, loss, xg_a: xgA, xg_b: xgB, likely_scores: likely,
    tie: (tieProbability(gap, models.early, home, true, maxGoals) + tieProbability(gap, models.early, home, false, maxGoals)) / 2,
    final: finalWin + finalDraw * extraTimeWin(gap, models.late, maxGoals),
  };
}

/** What /api/forecast/h2h would send, from data/matchups.json. Null for a team-season with no rating. */
export function playMatchup(matchups: Matchups, a: string, b: string, venue: Venue): HeadToHead | null {
  const sideA = matchups.sides[a], sideB = matchups.sides[b];
  if (!sideA || !sideB) return null;
  const strip = ({ exact: _exact, ...shown }: HeadToHeadSide & { exact: number }): HeadToHeadSide => shown;
  return { a: strip(sideA), b: strip(sideB), title_label: matchups.title_label,
    ...headToHead(sideA.exact, sideB.exact, matchups.models, matchups.home, venue, matchups.max_goals) };
}
