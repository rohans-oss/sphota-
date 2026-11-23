"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getExplain } from "@/lib/api";
import { useJobId } from "@/lib/useDone";
import { useJob } from "@/lib/store";
import type { ExplainResult, Mode } from "@/lib/types";
import { DecisionTree } from "@/components/app/DecisionTree";
import { DecodeNumber } from "@/components/motion/DecodeNumber";
import { Sheet } from "@/components/ui/BottomSheet";

const MODES: Mode[] = ["exact", "balanced", "small", "tiny"];

export default function ExplainPage() {
  const { id, ready } = useJobId();
  const router = useRouter();
  const set = useJob((s) => s.set);
  const used = useJob((s) => s.result?.mode) ?? "small";
  const [data, setData] = useState<ExplainResult>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (!id) return;
    getExplain(id).then(setData).catch((e) => setError(e instanceof Error ? e.message : "Could not load the explanation. Check that the backend is running."));
  }, [id]);

  if (!ready) return null;
  if (!id) {
    return (
      <div className="max-w-[560px]">
        <h1 className="h2-serif">Nothing to <em>explain yet.</em></h1>
        <p className="mt-4 text-gold-200">Compress a photo first. This page then shows why the system chose its mode.</p>
        <a href="/compress" className="btn-primary mt-6">Compress a photo</a>
      </div>
    );
  }

  function override(m: Mode) {
    set({ presetMode: m });
    router.push("/compress");
  }

  return (
    <div>
      <p className="eyebrow">Explain · why this mode</p>
      <h1 className="h2-serif mt-3">Every course, <em>one decision.</em></h1>
      {error && <p role="alert" className="mt-4 text-sm text-rust">{error}</p>}
      {data && (
        <div className="mt-8 grid gap-8 xl:grid-cols-[minmax(0,860px)_1fr]">
          <section className="panel p-5" aria-label="Decision tree">
            <p className="lab mb-2">Decision tree · the taken path is amber</p>
            <DecisionTree nodes={data.tree.nodes} edges={data.tree.edges} />
          </section>

          <div className="space-y-8">
            <section aria-label="Prediction">
              <p className="lab mb-3">XGBoost prediction vs actual</p>
              <dl className="grid grid-cols-2 gap-5">
                <div className="border-t border-gold-200/20 pt-3">
                  <dt className="lab">Size</dt>
                  <dd className="kpi mt-1 text-[34px]"><DecodeNumber value={data.predicted.kb.toFixed(1) + " KB"} /></dd>
                  <p className="mt-1 text-xs text-gold-400">± {data.predicted.kb_std.toFixed(1)} · actual {data.actual.kb.toFixed(1)} KB</p>
                </div>
                <div className="border-t border-gold-200/20 pt-3">
                  <dt className="lab">LPIPS</dt>
                  <dd className="kpi mt-1 text-[34px]"><DecodeNumber value={data.predicted.lpips.toFixed(2)} /></dd>
                  <p className="mt-1 text-xs text-gold-400">actual {data.actual.lpips.toFixed(2)}</p>
                </div>
              </dl>
            </section>

            <section aria-label="Rules">
              <p className="lab mb-3">Association rules that fired</p>
              <Sheet title="Rules" openLabel="Show rules">
                <ul className="space-y-3">
                  {data.rules.map((r, i) => (
                    <li key={i} className={"rounded-2xl border p-4 " + (r.fired ? "border-amber-400 bg-mocha-900" : "border-gold-200/15")}>
                      <p className="font-mono text-sm">{`{${r.antecedents.join(", ")}} → ${r.consequent}`}</p>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        <span className="chip">support {r.support.toFixed(2)}</span>
                        <span className="chip">confidence {r.confidence.toFixed(2)}</span>
                        <span className="chip">lift {r.lift.toFixed(1)}</span>
                      </div>
                      {r.fired && r.note && <p className="mt-2 text-sm text-amber-400">{r.note}</p>}
                    </li>
                  ))}
                </ul>
              </Sheet>
            </section>

            <section aria-label="Neighbours">
              <p className="lab mb-3">Nearest photos (kNN)</p>
              <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4 xl:grid-cols-2">
                {data.neighbours.map((n) => (
                  <li key={n.name}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={n.thumb_url} alt={n.name} className="aspect-[86/64] w-full rounded-lg border border-gold-200/20 object-cover" />
                    <p className="mt-1 text-sm">{n.name} <span className="lab !text-[10px]">· {n.best_mode}</span></p>
                  </li>
                ))}
              </ul>
            </section>

            <section aria-label="Override">
              <p className="lab mb-3">Override mode and re-encode</p>
              <div className="flex flex-wrap gap-2">
                {MODES.map((m) => (
                  <button key={m} className="chip min-h-11 px-4 capitalize" data-active={m === used} onClick={() => override(m)}>{m}</button>
                ))}
              </div>
            </section>
            <p className="text-xs text-gold-400">Demo values until the backend returns the real tree, rules and neighbours.</p>
          </div>
        </div>
      )}
    </div>
  );
}
