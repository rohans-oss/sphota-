"use client";
import { useRef } from "react";
import { gsap, useGSAP } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";

// A retro dial: the needle sweeps to the value with a small overshoot (guide F18). value is 0..1.
export function Gauge({ value }: { value: number }) {
  const needle = useRef<SVGLineElement>(null), arc = useRef<SVGPathElement>(null);
  useGSAP(
    () => {
      if (prefersReduced()) {
        gsap.set(needle.current, { rotation: 180 * value, svgOrigin: "95 100" });
        gsap.set(arc.current, { drawSVG: value * 100 + "%" });
        return;
      }
      gsap.fromTo(needle.current, { rotation: 0, svgOrigin: "95 100" }, { rotation: 180 * value, svgOrigin: "95 100", duration: 1.4, ease: "back.out(1.4)" });
      gsap.fromTo(arc.current, { drawSVG: "0%" }, { drawSVG: value * 100 + "%", duration: 1.4 });
    },
    { dependencies: [value] },
  );
  return (
    <svg viewBox="0 0 190 112" className="w-[190px]" role="img" aria-label={`Bin consistency ${(value * 100).toFixed(1)} percent`}>
      <path d="M15 100 A80 80 0 0 1 175 100" fill="none" stroke="#4A2F22" strokeWidth={14} strokeLinecap="round" />
      <path ref={arc} d="M15 100 A80 80 0 0 1 175 100" fill="none" stroke="#E07A2E" strokeWidth={14} strokeLinecap="round" />
      {Array.from({ length: 11 }, (_, i) => {
        const a = Math.PI * (1 - i / 10);
        return <line key={i} x1={95 + Math.cos(a) * 62} y1={100 - Math.sin(a) * 62} x2={95 + Math.cos(a) * 54} y2={100 - Math.sin(a) * 54} stroke="#BFA57A" strokeWidth={1.5} />;
      })}
      <line ref={needle} x1="95" y1="100" x2="17" y2="100" stroke="#F5E7C6" strokeWidth={3} strokeLinecap="round" />
      <circle cx="95" cy="100" r="7" fill="#F5E7C6" />
    </svg>
  );
}
