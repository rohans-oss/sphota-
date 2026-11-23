"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { getCompare, requestDecode, watchDecode } from "@/lib/api";
import { isDemo } from "@/lib/demo";
import { useJobId } from "@/lib/useDone";
import type { CompareResult } from "@/lib/types";
import { Crt } from "@/components/retro/Crt";
import { Segmented } from "@/components/ui/Segmented";
import { DecodeNumber } from "@/components/motion/DecodeNumber";
import { SplitView } from "@/components/landing/SplitView";

const CFGS = [["1", "1.0"], ["1.5", "1.5"], ["2", "2.0"], ["3", "3.0"]] as const;
const STEPS = [["10", "10"], ["25", "25"], ["50", "50"]] as const;

type Screen = { key: "jpeg" | "coarse" | "plain" | "sphota"; title: string; diffusion: boolean };
const SCREENS: Screen[] = [
  { key: "jpeg", title: "JPEG", diffusion: false }, { key: "coarse", title: "VAE only", diffusion: false },
  { key: "plain", title: "Plain diffusion", diffusion: true }, { key: "sphota", title: "Sphoṭa", diffusion: true },
];

const pct = (v: number | null) => (v == null ? "n/a" : Math.round(v * 100) + "%");
const num = (v: number | null) => (v == null ? "n/a" : v.toFixed(2));

export default function ComparePage() {
  const { id, ready } = useJobId();
  const [cfg, setCfg] = useState("1.5");
  const [steps, setSteps] = useState("25");
  const [data, setData] = useState<CompareResult>();
  const [preview, setPreview] = useState<{ url: string; step: number; total: number }>();
  const [split, setSplit] = useState(false);
  const [error, setError] = useState<string>();
  const stop = useRef<() => void>(() => {});

  const load = useCallback(async (jobId: string, c: string, s: string, withPreview: boolean) => {
    stop.current();
    try {
      if (withPreview) {
        const run = await requestDecode(jobId, +c, +s);
        await new Promise<void>((res, rej) => {
          stop.current = watchDecode(run.job_id, (e) => {
            if (e.stage === "denoise") setPreview({ url: e.preview_url, step: e.step, total: e.total });
            if (e.stage === "done") res();
            if (e.stage === "error") rej(new Error(e.message));
          });
        });
      }
      setData(await getCompare(jobId, +c, +s));
      setError(undefined);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load the comparison. Check that the backend is running.");
    } finally {
      setPreview(undefined);
    }
  }, []);

  // The page loads once per (job, guidance, steps) choice; `data` is deliberately read only to decide whether to stream previews.
  /* eslint-disable react-hooks/exhaustive-deps, react-hooks/set-state-in-effect */
  useEffect(() => {
    if (id) load(id, cfg, steps, !!data);
    return () => stop.current();
  }, [id, cfg, steps]);
  /* eslint-enable react-hooks/exhaustive-deps, react-hooks/set-state-in-effect */

  function onMove(e: React.PointerEvent<HTMLDivElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 100, y = ((e.clientY - r.top) / r.height) * 100;
    document.querySelectorAll<HTMLElement>(".cmp-img").forEach((img) => { img.style.transformOrigin = x + "% " + y + "%"; img.style.transform = "scale(3)"; });
  }
  function onLeave() { document.querySelectorAll<HTMLElement>(".cmp-img").forEach((i) => (i.style.transform = "")); }

  if (!ready) return null;
  if (!id) return <Empty />;
  const m = data?.metrics;
  const rows: [string, string, string][] = m ? [
    ["Text faithfulness", pct(m.tfr.plain), pct(m.tfr.sphota)],
    ["Face identity", num(m.fir.plain), num(m.fir.sphota)],
    ["Bin consistency", (m.consistency.plain * 100).toFixed(1) + "%", (m.consistency.sphota * 100).toFixed(1) + "%"],
    ["Decode time", m.decode_s.plain.toFixed(1) + " s", m.decode_s.sphota.toFixed(1) + " s"],
  ] : [];

  return (
    <div>
      <p className="eyebrow">Compare · one seed, four decoders</p>
      <h1 className="h2-serif mt-3">Same seed. <em>Four answers.</em></h1>
      <div className="mt-6 flex flex-wrap items-center gap-x-8 gap-y-4">
        <div><p className="lab mb-2">Guidance scale</p><Segmented value={cfg} onChange={setCfg} options={CFGS} label="Guidance scale" /></div>
        <div><p className="lab mb-2">DDIM steps</p><Segmented value={steps} onChange={setSteps} options={STEPS} label="DDIM steps" /></div>
        <button className="btn-ghost ml-auto" onClick={() => setSplit(true)} disabled={!data}>Fullscreen split</button>
      </div>
      {error && <p role="alert" className="mt-4 text-sm text-rust">{error}</p>}

      <div className="mt-8 grid gap-8 lg:grid-cols-[minmax(0,860px)_1fr]">
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2" onPointerMove={onMove} onPointerLeave={onLeave}>
          {SCREENS.map((s) => {
            const d = data?.[s.key];
            const showPreview = preview && s.diffusion;
            return (
              <figure key={s.key}>
                <Crt accent={s.key === "sphota"} className="aspect-[420/262] w-full">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  {d && <img src={showPreview ? preview!.url : d.url} alt={`${s.title} decode`} className="cmp-img size-full object-cover" />}
                  {showPreview && <span className="chip absolute left-3 top-3 bg-mocha-950/80">DENOISING {preview!.step} / {preview!.total}</span>}
                </Crt>
                <figcaption className="mt-3">
                  <span className={"font-medium " + (s.key === "sphota" ? "text-amber-400" : "")}>{s.title}</span>
                  {d && <span className="ml-3 font-mono text-xs text-gold-400">LPIPS {d.lpips == null ? "n/a" : d.lpips.toFixed(2)}</span>}
                  {d && <p className={"text-sm " + (d.kept ? "text-olive" : "text-rust")}>{d.verdict}</p>}
                </figcaption>
              </figure>
            );
          })}
        </div>
        <div className="panel h-fit p-6">
          <p className="lab">Metrics · plain diffusion vs Sphoṭa</p>
          <dl className="mt-4 space-y-4">
            {rows.map(([k, plain, sph]) => (
              <div key={k} className="border-t border-gold-200/15 pt-3">
                <dt className="text-sm text-gold-400">{k}</dt>
                <dd className="mt-1 flex items-baseline gap-4">
                  <span className="font-mono text-lg text-gold-400">{plain}</span>
                  <span className="text-gold-400">→</span>
                  <DecodeNumber key={cfg + steps + k + sph} value={sph} className="kpi text-[30px]" />
                </dd>
              </div>
            ))}
          </dl>
          <p className="mt-5 text-xs text-gold-400">{isDemo() ? "Demo values." : "Measured on this photo. n/a = nothing of that kind to check."}</p>
        </div>
      </div>
      {split && data && <SplitView left={data.plain.url} right={data.sphota.url} onClose={() => setSplit(false)} />}
    </div>
  );
}

function Empty() {
  return (
    <div className="max-w-[560px]">
      <h1 className="h2-serif">Nothing to <em>compare yet.</em></h1>
      <p className="mt-4 text-gold-200">Compress a photo first, then come back to see four decoders side by side.</p>
      <a href="/compress" className="btn-primary mt-6">Compress a photo</a>
    </div>
  );
}
