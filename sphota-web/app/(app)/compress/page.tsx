"use client";
import { useEffect, useRef, useState } from "react";
import { ArrowDown } from "@phosphor-icons/react";
import { compress, watchJob } from "@/lib/api";
import { useJob } from "@/lib/store";
import { crtOn, gsap } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";
import type { Mode } from "@/lib/types";
import { Crt } from "@/components/retro/Crt";
import { Dropzone } from "@/components/app/Dropzone";
import { FitFrame } from "@/components/app/FitFrame";
import { LockedBoxes } from "@/components/app/RegionBoxes";
import { ModeControls } from "@/components/app/ModeControls";
import { Pipeline } from "@/components/app/Pipeline";
import { KpiStrip } from "@/components/app/KpiStrip";
import { Sheet } from "@/components/ui/BottomSheet";

const SCREEN = 820 / 540;

export default function CompressPage() {
  const { file, previewUrl, result, stages, presetMode, set, mark } = useJob();
  const [mode, setMode] = useState<Mode>(presetMode ?? "small");
  const [network, setNetwork] = useState("2G");
  const [targetKb, setTargetKb] = useState(2);
  const [error, setError] = useState<string>();
  const [busy, setBusy] = useState(false);
  const screen = useRef<HTMLDivElement>(null);
  const stop = useRef<() => void>(() => {});

  useEffect(() => () => stop.current(), []);

  function pick(f: File) {
    setError(undefined);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    useJob.getState().reset();
    set({ file: f, previewUrl: URL.createObjectURL(f) });
    if (screen.current && !prefersReduced()) crtOn(screen.current);
  }

  async function run() {
    if (!file) return;
    setError(undefined); setBusy(true);
    useJob.getState().reset();
    try {
      const r = await compress(file, { mode, network, targetKb });
      set({ result: r });
      stop.current = watchJob(r.job_id, (e) => {
        if (e.stage === "denoise") { set({ denoise: { step: e.step, total: e.total, preview_url: e.preview_url } }); return; }
        if (e.stage === "error") { setError(e.message); setBusy(false); return; }
        if (e.stage === "done") { set({ done: e }); setBusy(false); return; }
        mark(e.stage, e.status, e.ms);
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Compression failed. Try again.");
      setBusy(false);
    }
  }

  const doneCount = Object.values(stages).filter((s) => s.status === "done").length;
  const current = Object.entries(stages).find(([, s]) => s.status === "running")?.[0];
  const rec = result?.recommended_mode ?? undefined;

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,820px)_1fr]">
      <section aria-label="Photo">
        <p className="lab mb-3">
          {file ? `${file.name} · ${(file.size / 1000).toFixed(0)} KB` : "No photo chosen"}
          {result ? ` · ${result.boxes.length} regions found` : ""}
        </p>
        <div ref={screen}>
          <Crt className="aspect-[4/5] w-full sm:aspect-[820/540]">
            {previewUrl ? (
              <FitFrame src={previewUrl} alt="Your photo" screenRatio={SCREEN}>
                {result && <LockedBoxes boxes={result.boxes} />}
              </FitFrame>
            ) : (
              <Dropzone onFile={pick} onError={setError} />
            )}
            {(busy || current) && (
              <div className="absolute inset-x-0 bottom-0 flex items-center gap-2 bg-mocha-950/80 px-4 py-2 font-mono text-xs tracking-wider text-gold-100">
                <span className="rec-dot text-rust">◉</span> REC · {current ?? "starting"} {Math.round((doneCount / 5) * 100)}%
              </div>
            )}
          </Crt>
        </div>
        {previewUrl && !busy && (
          <button className="btn-ghost mt-4 min-h-11" onClick={() => { useJob.getState().reset(); set({ file: undefined, previewUrl: undefined }); }}>
            Choose another photo
          </button>
        )}
        <KpiStrip result={result} />
      </section>

      <aside className="panel h-fit p-7" aria-label="Controls">
        <h1 className="h2-serif">Fold it into <em>a seed.</em></h1>
        <div className="mt-7">
          <ModeControls mode={mode} setMode={setMode} network={network} setNetwork={setNetwork} targetKb={targetKb} setTargetKb={setTargetKb} recommended={rec} />
        </div>
        <button className="btn-action mt-7 w-full" onClick={run} disabled={!file || busy}>
          {busy ? "Folding…" : "Compress to seed"}
          <ArrowDown size={18} weight="bold" className={busy ? "animate-bounce" : ""} />
        </button>
        {error && <p role="alert" className="mt-3 text-sm text-rust">{error}</p>}
        <div className="mt-7">
          <p className="lab mb-2">Pipeline</p>
          <Sheet title="Pipeline" openLabel="Show pipeline"><Pipeline /></Sheet>
        </div>
        {isDoneHint(result, busy) && <p className="mt-4 text-sm text-gold-200">Seed ready. Open Compare, Trust or Explain to see what it did.</p>}
      </aside>
    </div>
  );
}

const isDoneHint = (result: unknown, busy: boolean) => !!result && !busy;

// keep gsap referenced so tree-shaking keeps the plugin registrations from lib/motion
void gsap;
