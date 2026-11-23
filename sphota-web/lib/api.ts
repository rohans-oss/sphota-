import { isDemo, demoCompress, demoDenoise, demoEvents, demoGet } from "./demo";
import type { CompareResult, CompressResult, ExplainResult, JobEvent } from "./types";

// Empty = same origin: next.config.ts proxies /api/* to the Python service (SPHOTA_API, default http://127.0.0.1:8000).
const API = process.env.NEXT_PUBLIC_API ?? "";

// Throws the server's own explanation (FastAPI sends it as {"detail": ...}) so the pages can show why a call failed.
async function json<T>(res: Response, fallback: string): Promise<T> {
  if (!res.ok) {
    const d = await res.json().catch(() => ({}));
    throw new Error(typeof d.detail === "string" ? d.detail : fallback);
  }
  return (await res.json()) as T;
}
const network = (e: unknown) => (e instanceof TypeError ? new Error("The Sphota server is not reachable. Start it with: python -m truesight.app.api") : e);

export async function compress(file: File, o: { mode: string; network: string; targetKb: number }): Promise<CompressResult> {
  if (isDemo()) return demoCompress();
  const body = new FormData();
  body.append("file", file);
  body.append("mode", o.mode);
  body.append("network", o.network);
  body.append("target_kb", String(o.targetKb));
  try {
    return await json<CompressResult>(await fetch(API + "/api/compress", { method: "POST", body }), "Compression failed. Try again.");
  } catch (e) {
    throw network(e);
  }
}

export function watchJob(id: string, on: (e: JobEvent) => void) {
  if (isDemo()) return demoEvents(on);
  const es = new EventSource(API + "/api/jobs/" + id + "/events");
  let finished = false;
  es.onmessage = (m) => {
    const e = JSON.parse(m.data) as JobEvent;
    if (e.stage === "done" || e.stage === "error") finished = true;
    on(e);
    if (finished) es.close();
  };
  es.onerror = () => {
    es.close();
    if (!finished) on({ stage: "error", message: "Lost the connection to the Sphota server." });
  };
  return () => { finished = true; es.close(); };
}

export const getCompare = (id: string, cfg: number, steps: number): Promise<CompareResult> =>
  isDemo() ? demoGet("compare") : fetch(`${API}/api/compare/${id}?cfg=${cfg}&steps=${steps}`).then((r) => json<CompareResult>(r, "Could not load the comparison."));

export const getExplain = (id: string): Promise<ExplainResult> =>
  isDemo() ? demoGet("explain") : fetch(`${API}/api/explain/${id}`).then((r) => json<ExplainResult>(r, "Could not load the explanation."));

// Starts a new decode of the same file; the progress of THAT run is streamed from the returned job id.
export const requestDecode = (id: string, cfg: number, steps: number, seed = 7, safe_mode = false): Promise<{ job_id: string }> =>
  isDemo()
    ? Promise.resolve({ job_id: id })
    : fetch(`${API}/api/decode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: id, cfg, steps, seed, safe_mode }),
      }).then((r) => json<{ job_id: string }>(r, "Could not start the decode."));

// Live denoise previews while the diffusion screens re-decode.
export function watchDecode(id: string, on: (e: JobEvent) => void) {
  if (isDemo()) return demoDenoise(on);
  return watchJob(id, on);
}
