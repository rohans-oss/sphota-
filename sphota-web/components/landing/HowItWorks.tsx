"use client";
import { useEffect, useRef, useState } from "react";
import { gsap, useGSAP } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";
import { Crt } from "@/components/retro/Crt";
import { Scope } from "./Scope";

const STAGES = [
  { n: "01", t: "Encode", d: "A VAE folds the photo into a small latent grid.", chips: ["VAE", "Autoencoder"] },
  { n: "02", t: "Entropy-code", d: "A Gaussian mixture predicts each value, so bits go only to surprises.", chips: ["MLE", "GMM", "Hyperprior", "Arithmetic coding"] },
  { n: "03", t: "Transmit", d: "Only the 1.9 KB seed travels.", chips: ["Rate control"] },
  { n: "04", t: "Burst back", d: "Guided diffusion grows the photo; protected regions stay exact.", chips: ["Diffusion", "CFG", "Guidance", "ROI"] },
];

// The section pins while four stages play (guide F11). Under 1024 px or with reduced motion nothing pins:
// the four stages stack, each with its own static CRT picture.
export function HowItWorks() {
  const wrap = useRef<HTMLElement>(null);
  const [pos, setPos] = useState(0); // 0..4, fractional
  const [stacked, setStacked] = useState(false);

  useEffect(() => {
    const check = () => setStacked(window.innerWidth < 1024 || prefersReduced());
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  useGSAP(
    () => {
      gsap.matchMedia().add("(min-width: 1024px) and (prefers-reduced-motion: no-preference)", () => {
        const s = { v: 0 };
        gsap.to(s, {
          v: 4, ease: "none", onUpdate: () => setPos(s.v),
          scrollTrigger: { trigger: wrap.current, start: "top top", end: "+=150%", pin: true, scrub: 0.8 },
        });
      });
    },
    { scope: wrap },
  );

  const active = Math.min(3, Math.floor(pos));
  const local = Math.min(1, pos - active); // progress inside the active stage

  if (stacked) {
    return (
      <section id="how" ref={wrap} className="px-6 py-20 md:px-14">
        <p className="eyebrow">How it works · transmission log</p>
        <h2 className="h2-serif mt-3">Four stages, <em>one seed.</em></h2>
        <ol className="mt-10 grid gap-6 md:grid-cols-2">
          {STAGES.map((s, i) => (
            <li key={s.n} className="rounded-2xl bg-mocha-900 p-5">
              <Crt className="aspect-[16/10] w-full"><Scope stage={i} progress={1} /></Crt>
              <div className="mt-4 flex gap-4">
                <span className="w-11 font-logo text-[28px] text-amber-400">{s.n}</span>
                <div>
                  <h3 className="text-xl font-medium">{s.t}</h3>
                  <p className="mt-1 text-sm text-gold-400">{s.d}</p>
                  <div className="mt-2.5 flex flex-wrap gap-1.5">{s.chips.map((c) => <span key={c} className="chip">{c}</span>)}</div>
                </div>
              </div>
            </li>
          ))}
        </ol>
      </section>
    );
  }

  return (
    <section id="how" ref={wrap} className="relative grid min-h-[100svh] grid-cols-[560px_1fr] gap-16 px-14 pt-20">
      <div>
        <p className="eyebrow">How it works · transmission log</p>
        <h2 className="h2-serif mt-3">Four stages, <em>one seed.</em></h2>
        <ol className="mt-10 space-y-2">
          {STAGES.map((s, i) => (
            <li
              key={s.n}
              className={"flex gap-5 rounded-2xl px-5 py-3.5 transition-[opacity,background-color] duration-500 " + (i === active ? "bg-mocha-900 outline outline-1 outline-amber-400/50" : i < active ? "" : "opacity-45")}
            >
              <span className={"w-11 font-logo text-[28px] " + (i < active ? "text-olive" : i === active ? "text-amber-400" : "text-gold-400")}>
                {i < active ? "✓" : s.n}
              </span>
              <div>
                <h3 className="text-xl font-medium">{s.t}</h3>
                <p className="mt-1 text-sm text-gold-400">{s.d}</p>
                {i === active && (
                  <>
                    <div className="mt-2.5 flex flex-wrap gap-1.5">{s.chips.map((c) => <span key={c} className="chip">{c}</span>)}</div>
                    <div className="mt-3 h-[3px] w-[420px] rounded bg-mocha-600"><div className="h-full rounded bg-amber-400" style={{ width: local * 100 + "%" }} /></div>
                  </>
                )}
              </div>
            </li>
          ))}
        </ol>
      </div>
      <Crt className="mt-24 h-[470px] w-[700px]"><Scope stage={active} progress={local} /></Crt>
    </section>
  );
}
