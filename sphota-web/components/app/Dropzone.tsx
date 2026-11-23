"use client";
import { useRef, useState } from "react";

const BARS = ["#F5E7C6", "#E9D4A4", "#E9B66A", "#E07A2E", "#C9561E", "#86593F", "#4A2F22"];
const SAMPLES = [["medicine", "Medicine box"], ["invoice", "Invoice"], ["landscape", "Landscape"]] as const;
export const MAX_MB = 20;

// Test-card look: dashed gold border, a drop target and three sample photos (guide F16).
export function Dropzone({ onFile, onError }: { onFile: (f: File) => void; onError: (m: string) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  function accept(f: File | undefined) {
    if (!f) return;
    if (!/^image\/(png|jpeg|webp)$/.test(f.type)) return onError("That file type is not supported. Use PNG, JPEG or WebP.");
    if (f.size > MAX_MB * 1024 * 1024) return onError(`That file is ${(f.size / 1048576).toFixed(0)} MB. The limit is ${MAX_MB} MB.`);
    onFile(f);
  }
  async function sample(id: string) {
    const blob = await (await fetch(`/samples/${id}.webp`)).blob();
    accept(new File([blob], `${id}.webp`, { type: "image/webp" }));
  }

  return (
    <div
      className={"flex size-full flex-col items-center justify-center gap-3 border-2 border-dashed p-3 text-center sm:gap-5 sm:p-6 transition-colors " + (over ? "border-amber-400 bg-amber-400/10" : "border-gold-200/60")}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); accept(e.dataTransfer.files[0]); }}
    >
      <div className="flex h-6 w-44 overflow-hidden rounded sm:h-8 sm:w-56">{BARS.map((c) => <div key={c} className="flex-1" style={{ background: c }} />)}</div>
      <p className="font-display text-xl sm:text-2xl" style={{ fontVariationSettings: '"SOFT" 100' }}>Drop a photo or choose a sample</p>
      <button type="button" className="btn-primary" onClick={() => input.current?.click()}>Choose a file</button>
      <input ref={input} type="file" accept="image/png,image/jpeg,image/webp" className="sr-only" aria-label="Choose a photo" onChange={(e) => accept(e.target.files?.[0])} />
      <div className="flex flex-wrap justify-center gap-3">
        {SAMPLES.map(([id, label]) => (
          <button key={id} type="button" onClick={() => sample(id)} className="group w-[72px] text-left sm:w-24" aria-label={`Use sample: ${label}`}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`/samples/${id}.webp`} alt="" className="aspect-[4/3] w-full rounded-lg border border-gold-200/30 object-cover transition-transform group-hover:scale-105" />
            <span className="lab mt-1 block !text-[10px]">{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
