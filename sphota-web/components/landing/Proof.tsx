"use client";
import { useState } from "react";
import { Crt } from "@/components/retro/Crt";
import { RegionBox } from "@/components/app/RegionBoxes";
import { Segmented } from "@/components/ui/Segmented";
import { DecodeNumber } from "@/components/motion/DecodeNumber";
import { Reveal } from "@/components/motion/Reveal";
import { SplitView } from "./SplitView";

type Left = "jpeg" | "plain" | "coarse";
// Fractions of the image; generated with the demo photo (public/demo/boxes.json).
const DOSE = { x: 0.157, y: 0.468, w: 0.567, h: 0.218 };

const METRICS: [string, string, string][] = [
  ["Dose on the box", "500 mg", "plain diffusion: 800 mg"],
  ["Text kept exact", "98%", "plain diffusion: 46%"],
  ["Face identity", "0.91", "plain diffusion: 0.52"],
  ["LPIPS", "0.21", "plain diffusion: 0.24"],
];

// Same seed, same 1.9 KB: the left decoder changes, the right one is always Sphoṭa (guide F12).
export function Proof() {
  const [left, setLeft] = useState<Left>("plain");
  const [open, setOpen] = useState(false);
  const name = { jpeg: "JPEG", plain: "plain diffusion", coarse: "VAE only" }[left];
  return (
    <section id="proof" className="px-6 py-20 md:px-14">
      <Reveal>
        <p className="eyebrow">The proof · same seed, same 1.9 KB</p>
        <h2 className="h2-serif mt-3 max-w-[760px]">One dose changes. <em>One doesn&apos;t.</em></h2>
      </Reveal>
      <div className="mt-8"><Segmented value={left} onChange={setLeft} label="Left decoder" options={[["jpeg", "JPEG"], ["plain", "Plain diffusion"], ["coarse", "VAE only"]] as const} /></div>
      <div className="mt-8 grid gap-10 lg:grid-cols-[1fr_300px]">
        <div className="grid gap-6 md:grid-cols-2">
          <button onClick={() => setOpen(true)} aria-label="Open fullscreen comparison" className="block text-left">
            <Crt className="aspect-[4/3] w-full">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`/proof/${left}.webp`} alt={`Decoded with ${name} at 1.9 KB`} className="size-full object-cover" />
              {left === "plain" && <RegionBox box={DOSE} tone="rust" label="Changed · 800 mg" />}
            </Crt>
            <p className="lab mt-3">{name}</p>
          </button>
          <button onClick={() => setOpen(true)} aria-label="Open fullscreen comparison" className="block text-left">
            <Crt accent className="aspect-[4/3] w-full">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/proof/sphota.webp" alt="Decoded with Sphoṭa at 1.9 KB" className="size-full object-cover" />
              <RegionBox box={DOSE} tone="olive" label="Protected · 500 mg" />
            </Crt>
            <p className="lab mt-3 !text-amber-400">Sphoṭa</p>
          </button>
        </div>
        <dl className="space-y-5">
          {METRICS.map(([k, v, note]) => (
            <div key={k} className="border-t border-gold-200/20 pt-3">
              <dt className="lab">{k}</dt>
              <dd className="kpi mt-1 text-[40px]"><DecodeNumber value={v} /></dd>
              <p className="mt-1 text-xs text-gold-400">{note}</p>
            </div>
          ))}
          <p className="text-xs text-gold-400">Demo values. Replace with your evaluation results.</p>
        </dl>
      </div>
      {open && <SplitView left={`/proof/${left}.webp`} right="/proof/sphota.webp" onClose={() => setOpen(false)} />}
    </section>
  );
}
