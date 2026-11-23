// The whole app runs from files in public/demo, with realistic delays (guide F15).
// Demo values only: replace with real backend responses / evaluation results.
const fromQuery = () => {
  if (typeof window === "undefined") return false;
  const q = new URLSearchParams(window.location.search).get("demo");
  if (q === "0") { // ?demo=0 leaves a demo session that was remembered earlier
    try { sessionStorage.removeItem("demo"); } catch {}
    return false;
  }
  if (q !== null) {
    try { sessionStorage.setItem("demo", "1"); } catch {}
    return true;
  }
  try { return sessionStorage.getItem("demo") === "1"; } catch { return false; }
};
// A function, not a constant: ?demo=1 is remembered for the session so links between pages keep demo mode.
export const isDemo = () => process.env.NEXT_PUBLIC_DEMO === "1" || fromQuery();
export const DEMO = process.env.NEXT_PUBLIC_DEMO === "1";

import type { CompressResult, JobEvent } from "./types";

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function demoCompress(): Promise<CompressResult> {
  await wait(400);
  return (await fetch("/demo/compress.json")).json();
}
export async function demoGet(name: string) {
  await wait(600);
  return (await fetch(`/demo/${name}.json`)).json();
}
export function demoEvents(on: (e: JobEvent) => void) {
  let stop = false;
  (async () => {
    const events: JobEvent[] = await (await fetch("/demo/events.json")).json();
    for (const e of events) {
      if (stop) return;
      await wait(e.delay ?? 300);
      if (stop) return;
      on(e);
    }
  })();
  return () => {
    stop = true;
  };
}

// Streamed denoise preview frames for the Compare page (demo): noisy to sharp in about 1.5 s.
const DONE_STUB: JobEvent = {
  stage: "done", image_url: "/proof/sphota.webp", coarse_url: "/proof/coarse.webp",
  uncertainty_url: "/demo/uncertainty.png", consistency: 0.982, flagged: [],
};

export function demoDenoise(on: (e: JobEvent) => void) {
  let stop = false;
  (async () => {
    for (let i = 1; i <= 5; i++) {
      await wait(300);
      if (stop) return;
      on({ stage: "denoise", step: i * 5, total: 25, preview_url: `/demo/preview/denoise_${i}.webp` });
    }
    await wait(250);
    if (!stop) on(DONE_STUB);
  })();
  return () => { stop = true; };
}
