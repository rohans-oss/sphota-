"use client";
import { useEffect, useState, useSyncExternalStore } from "react";
import { isDemo } from "./demo";
import { useJob } from "./store";
import type { DoneEvent } from "./types";

// Mount-safe hooks: nothing here reads window while the server HTML is being matched, so hydration never differs.
// `ready` is true once the client has mounted (and, in demo mode, once the demo fallback has loaded), so pages
// never flash an empty state.
const noopSubscribe = () => () => {};
export const useMounted = () => useSyncExternalStore(noopSubscribe, () => true, () => false);

// The finished job from the store; in demo mode, falls back to the last demo event so Trust works on its own.
export function useDone(): { done?: DoneEvent; ready: boolean } {
  const done = useJob((s) => s.done);
  const mounted = useMounted();
  const [demo, setDemo] = useState<DoneEvent>();
  const [fetched, setFetched] = useState(false);
  const demoMode = mounted && isDemo();
  useEffect(() => {
    if (done || !demoMode) return;
    let alive = true;
    fetch("/demo/events.json")
      .then((r) => r.json())
      .then((ev: DoneEvent[]) => alive && setDemo(ev[ev.length - 1]))
      .finally(() => alive && setFetched(true));
    return () => { alive = false; };
  }, [done, demoMode]);
  return { done: done ?? demo, ready: mounted && (!!done || !demoMode || fetched) };
}

export function useJobId(): { id?: string; ready: boolean } {
  const id = useJob((s) => s.result?.job_id);
  const mounted = useMounted();
  return { id: id ?? (mounted && isDemo() ? "demo-001" : undefined), ready: mounted };
}
