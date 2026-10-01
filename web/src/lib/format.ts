// Number and time formatting shared by every screen. Figures use a true minus sign and en-US grouping.

export const DASH = "–";
const MINUS = "−";

export function num(value: number | null | undefined, decimals = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  const text = Math.abs(value).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  const zero = Number(text.replace(/,/g, "")) === 0;
  return value < 0 && !zero ? `${MINUS}${text}` : text;
}

export function signed(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  const text = num(Math.abs(value), decimals);
  if (Number(text.replace(/,/g, "")) === 0) return text;
  return value > 0 ? `+${text}` : `${MINUS}${text}`;
}

/** A 0–1 share as a percentage: 0.246 → "25%". */
export function share(value: number | null | undefined, decimals = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `${num(value * 100, decimals)}%`;
}

export function stat(value: number | null | undefined, meta: { decimals: number; percent: boolean }): string {
  const text = num(value, meta.decimals);
  return meta.percent && text !== DASH ? `${text}%` : text;
}

export function ordinal(n: number): string {
  const tens = n % 100;
  if (tens >= 11 && tens <= 13) return `${n}th`;
  return `${n}${({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th"}`;
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;
}

/** Elapsed seconds: "0.4 s", "12 s", "3 min 05 s". */
export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) return DASH;
  if (seconds < 10) return `${seconds.toFixed(1)} s`;
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${String(Math.round(seconds % 60)).padStart(2, "0")} s`;
}

export function clockTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleTimeString("en-GB", { hour12: false });
}

export function ago(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return DASH;
  const seconds = Math.max(0, (now - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
  return plural(Math.round(seconds / 86400), "day") + " ago";
}
