import { useEffect, useId, useRef, useState } from "react";
import { api } from "../api";
import type { TeamHit } from "../types";
import { SearchIcon } from "./Icons";

/** Typeahead over every team since 2011-12 (and the live season): arrows move, Enter picks, Escape closes. */
export function TeamSearch({ onPick, placeholder = "Search a club", initial = "", label = "Search teams" }: {
  onPick: (hit: TeamHit) => void; placeholder?: string; initial?: string; label?: string;
}) {
  const [query, setQuery] = useState(initial);
  const [hits, setHits] = useState<TeamHit[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => setQuery(initial), [initial]);
  useEffect(() => {
    if (query.trim().length < 2) {
      setHits([]);
      return;
    }
    let live = true;
    const timer = window.setTimeout(() => {
      api.teams(query).then((found) => { if (live) { setHits(found); setActive(0); } }, () => undefined);
    }, 120);
    return () => { live = false; window.clearTimeout(timer); };
  }, [query]);
  useEffect(() => {
    const close = (event: MouseEvent) => { if (!box.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const pick = (hit: TeamHit) => {
    setQuery(hit.name);
    setOpen(false);
    onPick(hit);
  };
  const showing = open && hits.length > 0;
  return (
    <div className="search" ref={box}>
      <SearchIcon />
      <input
        type="search" value={query} placeholder={placeholder} aria-label={label} autoComplete="off"
        role="combobox" aria-expanded={showing} aria-controls={listId}
        aria-activedescendant={showing ? `${listId}-${active}` : undefined}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={(e) => { e.target.select(); setOpen(true); }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setOpen(true); setActive((i) => Math.min(i + 1, hits.length - 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setActive((i) => Math.max(i - 1, 0)); }
          else if (e.key === "Enter" && showing) { e.preventDefault(); pick(hits[active]); }
          else if (e.key === "Escape") setOpen(false);
        }}
      />
      {showing && (
        <ul className="search-list" id={listId} role="listbox">
          {hits.map((hit, i) => (
            <li key={hit.team_id} id={`${listId}-${i}`} role="option" aria-selected={i === active}
                onMouseEnter={() => setActive(i)} onMouseDown={(e) => { e.preventDefault(); pick(hit); }}>
              <span className="name">{hit.name}</span>
              <span className="meta">{hit.seasons.length === 1 ? hit.seasons[0].label
                : `${hit.seasons.length} seasons, latest ${hit.seasons[0].label}`}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
