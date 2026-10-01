import { describe, expect, it } from "vitest";
import { ago, duration, num, ordinal, plural, share, signed, stat } from "./format";

describe("format", () => {
  it("formats numbers with grouping, a true minus and a dash for missing values", () => {
    expect(num(1234.567, 1)).toBe("1,234.6");
    expect(num(-0.42, 2)).toBe("−0.42");
    expect(num(-0.001, 2)).toBe("0.00");
    expect(num(null)).toBe("–");
    expect(num(Number.NaN)).toBe("–");
  });
  it("signs differences", () => {
    expect(signed(0.78)).toBe("+0.78");
    expect(signed(-0.16)).toBe("−0.16");
    expect(signed(0)).toBe("0.00");
  });
  it("shows shares and stats with their units", () => {
    expect(share(0.246)).toBe("25%");
    expect(stat(64.1, { decimals: 1, percent: true })).toBe("64.1%");
    expect(stat(671, { decimals: 0, percent: false })).toBe("671");
    expect(stat(null, { decimals: 1, percent: true })).toBe("–");
  });
  it("writes ordinals and plurals", () => {
    expect([1, 2, 3, 4, 11, 12, 13, 21, 22, 103].map(ordinal)).toEqual(
      ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "103rd"]);
    expect(plural(1, "match", "matches")).toBe("1 match");
    expect(plural(1290, "minute")).toBe("1,290 minutes");
  });
  it("describes durations and ages", () => {
    expect(duration(0.42)).toBe("0.4 s");
    expect(duration(42.4)).toBe("42 s");
    expect(duration(185)).toBe("3 min 05 s");
    expect(duration(null)).toBe("–");
    const now = Date.parse("2026-10-01T12:00:00Z");
    expect(ago("2026-10-01T11:59:30Z", now)).toBe("just now");
    expect(ago("2026-10-01T11:00:00Z", now)).toBe("1 h ago");
    expect(ago("2026-09-28T12:00:00Z", now)).toBe("3 days ago");
  });
});
