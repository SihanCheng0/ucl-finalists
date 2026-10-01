import { useCallback, useEffect, useState } from "react";

export interface Loaded<T> { data: T | null; error: Error | null; loading: boolean; reload: () => void }

/** Load `load()` whenever `deps` change; the previous data stays visible while the next load runs. A null `load`
 * means there is nothing to load yet. */
export function useApi<T>(load: (() => Promise<T>) | null, deps: unknown[]): Loaded<T> {
  const [state, setState] = useState<{ data: T | null; error: Error | null; loading: boolean }>(
    { data: null, error: null, loading: load !== null });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (load === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    load().then(
      (data) => { if (live) setState({ data, error: null, loading: false }); },
      (error: Error) => { if (live) setState({ data: null, error, loading: false }); },
    );
    return () => { live = false; };
  }, [...deps, nonce]); // eslint-disable-line react-hooks/exhaustive-deps
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, reload };
}
