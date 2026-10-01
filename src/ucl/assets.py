"""The report page's stylesheet and tooltip script (artifact page contract, dataviz specs)."""

CSS = """
/* Layout: one 46rem reading column; each chart is a surface card that reflows by its own width. */
:root {
  --bg: #f3f5f8; --surface: #fcfcfd; --ink: #0d1422; --ink-2: #4a5263; --muted: #656b79;
  --hair: #e1e5eb; --rule: #c3c9d3; --accent: #23408e; --on-accent: #ffffff;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --neg: #e34948; --mark: #fde9ad;
  --good: #0ca30c; --warn: #fab219; --on-good: #ffffff; --on-warn: #3a2a00;
  --font-display: "Archivo", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-body: "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0b0f16; --surface: #131923; --ink: #f2f4f8; --ink-2: #b8bfcb; --muted: #8b919d;
    --hair: #252d3a; --rule: #3a4456; --accent: #a9bbff; --on-accent: #0b0f16;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neg: #e66767; --mark: #5c4a12;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0b0f16; --surface: #131923; --ink: #f2f4f8; --ink-2: #b8bfcb; --muted: #8b919d;
  --hair: #252d3a; --rule: #3a4456; --accent: #a9bbff; --on-accent: #0b0f16;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --neg: #e66767; --mark: #5c4a12;
  color-scheme: dark;
}
[hidden] { display: none !important; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 400 16px/1.6 var(--font-body); }
.page { max-width: 46rem; margin: 0 auto; padding-inline: 16px; padding-block: 48px 72px; display: grid; gap: 64px; }
section, .masthead { display: grid; gap: 16px; min-width: 0; }
h1, h2, h3 { margin: 0; font-family: var(--font-display); line-height: 1.12; text-wrap: balance; }
h1 { font-size: clamp(2.1rem, 1.5rem + 3vw, 3.3rem); font-weight: 800; font-stretch: 112%; letter-spacing: -0.015em; }
h2 { font-size: 1.6rem; font-weight: 750; font-stretch: 112%; }
h3 { font-size: 1.15rem; font-weight: 700; font-stretch: 106%; }
h4 { margin: 8px 0 0; font: 600 0.78rem/1.3 var(--font-body); letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-2); }
p { margin: 0; max-width: 65ch; }
code { padding: 1px 5px; font: 0.9em ui-monospace, "SF Mono", Menlo, monospace; background: var(--hair); border-radius: 4px; }
.eyebrow { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: var(--ink-2); }
.dek { font-size: 1.12rem; color: var(--ink-2); }
.notice { padding: 12px 16px; color: var(--ink-2); background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; }
.final-list { margin: 0; padding: 0; list-style: none; border-top: 2px solid var(--ink); }
.final-list li { display: grid; grid-template-columns: 5.5rem minmax(0, 1fr) auto; gap: 4px 16px; align-items: baseline; padding-block: 14px; border-bottom: 1px solid var(--hair); }
.final-list .season, .final-list .city { font-size: 0.88rem; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.final-list .tie { min-width: 0; }
.final-list .score { display: inline-block; margin-inline: 10px; font-family: var(--font-display); font-weight: 800; font-stretch: 125%; }
.final-list .pens { display: block; font-size: 0.82rem; color: var(--ink-2); }
.chart { container-type: inline-size; display: grid; gap: 12px; min-width: 0; padding: 16px; background: var(--surface); border: 1px solid var(--hair); border-radius: 10px; }
.chart h4 { margin: 0; }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); gap: 16px; min-width: 0; }
.bars, .dots { display: grid; gap: 2px; min-width: 0; }
.bar-row, .dot-row, .axis-row { display: grid; grid-template-columns: minmax(0, 14rem) minmax(0, 1fr) 7rem; gap: 4px 12px; align-items: center; }
.bar-row, .dot-row { min-height: 30px; margin-inline: -6px; padding: 3px 6px; border-radius: 6px; }
.bar-row:hover, .dot-row:hover { background: color-mix(in srgb, var(--ink) 5%, transparent); }
.bar-row:focus-visible, .dot-row:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.bar-label { min-width: 0; font-size: 0.86rem; overflow-wrap: anywhere; }
.bar-label small { display: block; font-size: 0.76rem; color: var(--muted); }
.bar-value { font-size: 0.84rem; color: var(--ink-2); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.note { display: block; font-size: 0.72rem; color: var(--muted); white-space: normal; }
.chip { display: inline-block; margin-left: 8px; padding: 0 7px; font-size: 0.7rem; line-height: 1.6; color: var(--ink-2); white-space: nowrap; border: 1px solid var(--rule); border-radius: 999px; }
.track { position: relative; display: flex; align-items: center; height: 14px; border-left: 1px solid var(--rule); }
.track .bar { display: block; width: var(--w); height: 14px; background: var(--c); border-radius: 0 4px 4px 0; }
.track.split { display: grid; grid-template-columns: 1fr 1fr; border-left: 0; }
.track .half { display: flex; height: 14px; }
.track .half.neg { justify-content: flex-end; border-right: 1px solid var(--rule); }
.track .half.neg .bar { border-radius: 4px 0 0 4px; }
.track.dots { height: 22px; border-left: 0; }
.track.dots::before { content: ""; position: absolute; inset: 50% 0 auto; border-top: 1px solid var(--rule); }
.track.dots .tick { position: absolute; left: var(--x); top: 4px; height: 14px; border-left: 2px solid var(--muted); transform: translateX(-1px); }
.track.dots .dot { position: absolute; left: var(--x); top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; background: var(--s1); border-radius: 50%; box-shadow: 0 0 0 2px var(--surface); }
.axis { position: relative; height: 18px; font-size: 0.72rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.axis span { position: absolute; left: var(--x); transform: translateX(-50%); }
.axis span:first-child { transform: none; }
.axis span:last-child { transform: translateX(-100%); }
.chart-caption { font-size: 0.78rem; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.8rem; color: var(--ink-2); }
.key { display: inline-flex; align-items: center; gap: 7px; }
.swatch { display: inline-block; flex: none; background: var(--c); }
.swatch.bar { width: 14px; height: 10px; border-radius: 0 3px 3px 0; }
.swatch.dot { width: 10px; height: 10px; border-radius: 50%; }
.swatch.tick { width: 2px; height: 12px; }
@container (max-width: 34rem) {
  .bar-row, .dot-row, .axis-row { grid-template-columns: minmax(0, 1fr) 6.5rem; }
  .bar-label { grid-column: 1 / -1; }
  .axis-row > span:first-child { display: none; }
}
.twin { min-width: 0; }
.twin summary { width: fit-content; font-size: 0.84rem; color: var(--accent); cursor: pointer; }
.twin summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
.table-wrap:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.table-wrap { max-width: 100%; margin-top: 8px; overflow-x: auto; background: var(--surface); border: 1px solid var(--hair); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th, td { padding: 7px 10px; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--hair); }
th { font-weight: 600; color: var(--ink-2); }
tr:last-child td { border-bottom: 0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.cards { display: grid; gap: 28px; min-width: 0; }
.card { display: grid; gap: 18px; min-width: 0; padding: 22px; background: var(--surface); border: 1px solid var(--hair); border-radius: 12px; }
.card .chart { padding: 0; background: none; border: 0; }
.card-head { display: grid; gap: 4px; min-width: 0; overflow-wrap: anywhere; }
.card-head h3 span { margin-left: 6px; font: 500 0.95rem/1 var(--font-body); color: var(--ink-2); }
.card-head p { font-size: 0.9rem; color: var(--ink-2); }
.ladder { position: relative; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 2px; padding-bottom: 12px; }
.rung { padding: 5px 2px; font-size: 0.72rem; text-align: center; color: var(--ink-2); background: var(--hair); }
.rung:first-child { border-radius: 6px 0 0 6px; }
.rung:nth-child(5) { border-radius: 0 6px 6px 0; }
.rung.reached { color: var(--ink); background: color-mix(in srgb, var(--s1) 22%, var(--surface)); }
.rung.here { font-weight: 600; color: var(--on-accent); background: var(--accent); }
.expect { position: absolute; left: var(--x); bottom: 0; width: 0; height: 0; transform: translateX(-50%); border: 6px solid transparent; border-top: 0; border-bottom: 8px solid var(--ink); }
.ladder-note { font-size: 0.82rem; color: var(--ink-2); }
.narrative, .ai-summary { display: grid; gap: 10px; min-width: 0; overflow-wrap: anywhere; }
.ai-summary { padding-top: 16px; border-top: 1px solid var(--hair); }
.narrative ul, .ai-summary ul { margin: 0; padding-left: 1.2em; }
.ai-missing { font-size: 0.84rem; color: var(--ink-2); }
.badge { display: inline-flex; align-items: center; gap: 7px; width: fit-content; font-size: 0.78rem; color: var(--ink-2); }
.badge .icon { display: inline-grid; place-items: center; width: 17px; height: 17px; font-size: 0.68rem; font-weight: 700; color: var(--surface); background: var(--muted); border-radius: 50%; }
.badge.good .icon { color: var(--on-good); background: var(--good); }
.badge.warn .icon { color: var(--on-warn); background: var(--warn); }
mark.unverified { padding-inline: 2px; color: inherit; background: var(--mark); border-radius: 3px; }
.method ul { display: grid; gap: 8px; max-width: 65ch; margin: 0; padding-left: 1.2em; }
footer { font-size: 0.8rem; color: var(--muted); }
.tip { position: fixed; z-index: 10; max-width: 19rem; padding: 8px 11px; font-size: 0.8rem; line-height: 1.45; white-space: pre-line; color: var(--bg); background: var(--ink); border-radius: 7px; pointer-events: none; box-shadow: 0 6px 18px rgb(0 0 0 / 0.2); }
.tip::first-line { font-weight: 700; }
@media (max-width: 560px) { .final-list li { grid-template-columns: 1fr; } }
@media (forced-colors: active) { .track .bar, .track.dots .dot, .track.dots .tick, .swatch, .rung { forced-color-adjust: none; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

JS = """
(() => {
  const tip = document.getElementById("tip");
  if (!tip) return;
  const place = (x, y) => {
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.max(8, Math.min(x + 14, window.innerWidth - w - 8)) + "px";
    tip.style.top = (y - h - 14 < 8 ? y + 18 : y - h - 14) + "px";
  };
  const show = (el, x, y) => {
    tip.textContent = el.getAttribute("data-tip");
    tip.hidden = false;
    place(x, y);
  };
  const owner = (e) => (e.target instanceof Element ? e.target.closest("[data-tip]") : null);
  document.addEventListener("pointermove", (e) => {
    const el = owner(e);
    if (el) show(el, e.clientX, e.clientY);
    else tip.hidden = true;
  });
  document.documentElement.addEventListener("pointerleave", () => { tip.hidden = true; });
  document.addEventListener("focusin", (e) => {
    const el = owner(e);
    if (!el) return;
    const r = el.getBoundingClientRect();
    show(el, r.left + Math.min(r.width / 2, 160), r.top);
  });
  document.addEventListener("focusout", () => { tip.hidden = true; });
  window.addEventListener("scroll", () => { tip.hidden = true; }, { passive: true });
})();
"""
