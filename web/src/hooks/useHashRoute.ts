import { useCallback, useEffect, useState } from "react";
import { formatHash, parseHash, type Route } from "../lib/route";

export function useHashRoute(): [Route, (route: Route) => void] {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = useCallback((next: Route) => {
    const hash = formatHash(next);
    if (hash !== window.location.hash) window.location.hash = hash;
  }, []);
  return [route, navigate];
}
