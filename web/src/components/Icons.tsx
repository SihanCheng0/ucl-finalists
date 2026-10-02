// Inline icons, drawn at 24 units with a 1.75 stroke so they sit with Public Sans.
import type { SVGProps } from "react";

const base = { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.75, strokeLinecap: "round",
  strokeLinejoin: "round", "aria-hidden": true } as const;

export const SearchIcon = (p: SVGProps<SVGSVGElement>) => <svg {...base} {...p}><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>;
export const CheckIcon = (p: SVGProps<SVGSVGElement>) => <svg {...base} {...p}><path d="m5 12.5 4.5 4.5L19 7.5" /></svg>;
export const CrossIcon = (p: SVGProps<SVGSVGElement>) => <svg {...base} {...p}><path d="M6 6l12 12M18 6 6 18" /></svg>;
export const AlertIcon = (p: SVGProps<SVGSVGElement>) => <svg {...base} {...p}><path d="M12 8v5M12 16.5v.5" /><path d="M10.3 3.9 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" /></svg>;
export const SkipIcon = (p: SVGProps<SVGSVGElement>) => <svg {...base} {...p}><path d="M6 12h12" /></svg>;
export const SwapIcon = (p: SVGProps<SVGSVGElement>) => <svg {...base} {...p}><path d="M7 7h11l-3-3M17 17H6l3 3" /></svg>;
export const SunIcon = (p: SVGProps<SVGSVGElement>) => <svg {...base} {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>;
export const MoonIcon = (p: SVGProps<SVGSVGElement>) => <svg {...base} {...p}><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z" /></svg>;
export const AutoIcon = (p: SVGProps<SVGSVGElement>) => <svg {...base} {...p}><circle cx="12" cy="12" r="8" /><path d="M12 4a8 8 0 0 0 0 16Z" fill="currentColor" /></svg>;
export const CloseIcon = CrossIcon;
export const InfoIcon = (p: SVGProps<SVGSVGElement>) => <svg {...base} {...p}><circle cx="12" cy="12" r="9" /><path d="M12 11v5.5M12 7.5v.5" /></svg>;

/** The starball, reduced to a ring of eight points: a nod to the competition, not its mark. */
export const Mark = (p: SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" aria-hidden {...p}>
    <circle cx="12" cy="12" r="10.5" fill="none" stroke="var(--accent)" strokeWidth="1.75" />
    {Array.from({ length: 8 }, (_, i) => {
      const a = (i / 8) * Math.PI * 2;
      return <circle key={i} cx={12 + Math.cos(a) * 5.6} cy={12 + Math.sin(a) * 5.6} r="1.5" fill="var(--accent)" />;
    })}
  </svg>
);
