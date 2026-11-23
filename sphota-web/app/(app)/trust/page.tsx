"use client";
import { useState } from "react";
import { DownloadSimple } from "@phosphor-icons/react";
import { useJob } from "@/lib/store";
import { useDone } from "@/lib/useDone";
import { Crt } from "@/components/retro/Crt";
import { FitFrame } from "@/components/app/FitFrame";
import { HeatOverlay } from "@/components/app/HeatOverlay";
import { RegionBox } from "@/components/app/RegionBoxes";
import { Gauge } from "@/components/app/Gauge";
import { FlagList } from "@/components/app/FlagList";
import { Segmented } from "@/components/ui/Segmented";
import { Toggle } from "@/components/ui/Toggle";
import { Sheet } from "@/components/ui/BottomSheet";
import { DecodeNumber } from "@/components/motion/DecodeNumber";

type View = "off" | "heat" | "boxes";

export default function TrustPage() {
  const { done, ready } = useDone();
  const result = useJob((s) => s.result);
  const [view, setView] = useState<View>("heat");
  const [safe, setSafe] = useState(false);
  const [hover, setHover] = useState<string>();

  if (!ready) return null;
  if (!done) {
    return (
      <div className="max-w-[560px]">
        <h1 className="h2-serif">No trust report <em>yet.</em></h1>
        <p className="mt-4 text-gold-200">Compress a photo first. The witness then shows where the decoder invented detail.</p>
        <a href="/compress" className="btn-primary mt-6">Compress a photo</a>
      </div>
    );
  }
  const locked = result?.boxes ?? [{ x: 0.157, y: 0.468, w: 0.567, h: 0.218, kind: "text" as const, label: "Dose 500 mg" }];

  return (
    <div>
      <p className="eyebrow">Trust · the witness</p>
      <h1 className="h2-serif mt-3">It knows what <em>it invented.</em></h1>
      <div className="mt-8 grid gap-8 lg:grid-cols-[minmax(0,820px)_1fr]">
        <section aria-label="Witness map">
          <Crt className="aspect-[820/540] w-full">
            <FitFrame src={done.image_url} alt="Decoded photo with uncertainty witness" screenRatio={820 / 540}>
              <HeatOverlay src={done.uncertainty_url} visible={view === "heat"} />
              {locked.map((b, i) => <RegionBox key={"l" + i} box={b} tone="olive" label={b.label} />)}
              {view === "boxes" && done.flagged.map((f) => (
                <RegionBox key={f.id} box={f.box} tone="rust" label={f.name} active={hover === f.id} onEnter={() => setHover(f.id)} onLeave={() => setHover(undefined)} />
              ))}
              {/* safe mode: the honest VAE decode fades in over each flagged region */}
              {done.flagged.map((f) => (
                <div key={"s" + f.id} className="pointer-events-none absolute overflow-hidden transition-opacity duration-500"
                  style={{ left: f.box.x * 100 + "%", top: f.box.y * 100 + "%", width: f.box.w * 100 + "%", height: f.box.h * 100 + "%", opacity: safe ? 1 : 0 }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={done.coarse_url} alt="" className="absolute max-w-none object-fill"
                    style={{ width: 100 / f.box.w + "%", height: 100 / f.box.h + "%", left: -(f.box.x / f.box.w) * 100 + "%", top: -(f.box.y / f.box.h) * 100 + "%" }} />
                </div>
              ))}
              {hover && view !== "boxes" && (() => {
                const f = done.flagged.find((x) => x.id === hover);
                return f ? <RegionBox box={f.box} tone="rust" label={f.name} active /> : null;
              })()}
            </FitFrame>
          </Crt>
          <div className="mt-5 flex flex-wrap items-center gap-x-8 gap-y-4">
            <Segmented value={view} onChange={setView} label="Witness view" options={[["off", "Off"], ["heat", "Heat"], ["boxes", "Boxes"]] as const} />
            <Toggle checked={safe} onChange={setSafe} label="Safe mode" />
            <button className="btn-ghost no-print ml-auto" onClick={() => window.print()}><DownloadSimple size={18} /> Download trust report</button>
          </div>
        </section>

        <aside className="space-y-6">
          <div className="panel flex items-center gap-6 p-6">
            <Gauge value={done.consistency} />
            <div>
              <p className="lab">Bin consistency</p>
              <DecodeNumber value={(done.consistency * 100).toFixed(1) + "%"} className="kpi text-[44px]" />
              <p className="mt-1 text-sm text-gold-400">Re-encoding the output gives the transmitted latent.</p>
            </div>
          </div>
          <div>
            <p className="lab mb-3">Flagged regions · {done.flagged.length}</p>
            <Sheet title="Flagged regions" openLabel={`Show ${done.flagged.length} flagged regions`}>
              <FlagList flags={done.flagged} hover={hover} setHover={setHover} />
            </Sheet>
          </div>
          <p className="text-xs text-gold-400">Demo values until the backend returns real uncertainty maps.</p>
        </aside>
      </div>
    </div>
  );
}
