"use client";
import { useRef } from "react";
import { gsap, useGSAP, crtOff, crtOn } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";
import { Crt } from "./Crt";

const BARS = ["#F5E7C6", "#E9D4A4", "#E9B66A", "#E07A2E", "#C9561E", "#86593F", "#4A2F22"];

// 404 test card: bars roll down twice, then the screen collapses to a line and powers back on, every 6 s (guide F21).
export function TestCard() {
  const root = useRef<HTMLDivElement>(null);
  useGSAP(
    () => {
      if (prefersReduced()) return;
      const tl = gsap.timeline({ repeat: -1, repeatDelay: 3.2, delay: 1.5 });
      tl.fromTo(".bars", { yPercent: 0 }, { yPercent: 100, duration: 0.5, ease: "none" })
        .fromTo(".bars", { yPercent: -100 }, { yPercent: 0, duration: 0.5, ease: "none" })
        .fromTo(".bars", { yPercent: 0 }, { yPercent: 100, duration: 0.5, ease: "none" })
        .fromTo(".bars", { yPercent: -100 }, { yPercent: 0, duration: 0.5, ease: "none" })
        .add(crtOff(".testcard"))
        .add(crtOn(".testcard", 0.2));
    },
    { scope: root },
  );
  return (
    <div ref={root}>
      <Crt tilt={-2} className="testcard h-[320px] w-full md:h-[420px]">
        <div className="flex h-full flex-col">
          <div className="relative h-2/3 overflow-hidden">
            <div className="bars flex h-full">{BARS.map((c) => <div key={c} className="flex-1" style={{ background: c }} />)}</div>
          </div>
          <div className="flex h-[10%]">{[...BARS].reverse().map((c) => <div key={c} className="flex-1 opacity-70" style={{ background: c }} />)}</div>
          <div className="grid flex-1 place-items-center font-logo text-[44px] text-amber-400">404</div>
        </div>
      </Crt>
    </div>
  );
}
