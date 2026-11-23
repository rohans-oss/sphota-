"use client";
import { useRef } from "react";
import { gsap, useGSAP } from "@/lib/motion";

const LINES = ["Other AI codecs make photos pretty.", "Sphoṭa makes them pretty and true.", "What you send is what bursts forth."];
const ACCENT = new Set(["and", "true."]);

// Words start dim and light up in reading order as you scroll (guide F10).
export function Tagline() {
  const ref = useRef<HTMLParagraphElement>(null);
  useGSAP(
    () => {
      gsap.matchMedia().add("(prefers-reduced-motion: no-preference)", () => {
        gsap.fromTo(".tw", { opacity: 0.22 }, {
          opacity: 1, stagger: 0.1, ease: "none",
          scrollTrigger: { trigger: ref.current, start: "top 75%", end: "bottom 45%", scrub: 0.6 },
        });
      });
    },
    { scope: ref },
  );
  return (
    <p
      ref={ref} className="px-6 py-24 font-display text-[clamp(34px,4.4vw,60px)] leading-[1.06] tracking-[-0.035em] md:px-14 md:py-32"
      style={{ fontVariationSettings: '"SOFT" 100' }}
    >
      {LINES.map((line, i) => (
        <span key={i} className="block">
          {line.split(" ").map((w, j) => (
            <span key={j} className={"tw mr-[0.25em] inline-block " + (i === 1 && ACCENT.has(w) ? "italic text-amber-400" : "")}>{w}</span>
          ))}
        </span>
      ))}
    </p>
  );
}
