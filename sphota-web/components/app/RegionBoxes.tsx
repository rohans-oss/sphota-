"use client";
import { useRef } from "react";
import { gsap, useGSAP } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";
import type { Box } from "@/lib/types";

type Frac = { x: number; y: number; w: number; h: number };

// Boxes use fractions of the image (0..1), so they stay aligned at every size (guide F12).
export function RegionBox({
  box, tone, label, active, onEnter, onLeave,
}: { box: Frac; tone: "olive" | "rust"; label: string; active?: boolean; onEnter?: () => void; onLeave?: () => void }) {
  const c = tone === "olive" ? "border-olive" : "border-rust";
  const b = tone === "olive" ? "bg-olive text-mocha-950" : "bg-rust text-gold-100";
  return (
    <div
      className={"region absolute rounded border-dashed " + c + (active ? " border-[4px] bg-gold-100/10" : " border-2")}
      style={{ left: box.x * 100 + "%", top: box.y * 100 + "%", width: box.w * 100 + "%", height: box.h * 100 + "%" }}
      onMouseEnter={onEnter} onMouseLeave={onLeave}
    >
      <span className={"absolute -top-6 left-0 rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-wider whitespace-nowrap " + b}>{label}</span>
    </div>
  );
}

// Locked regions drawn as detection returns, staggered 60 ms.
export function LockedBoxes({ boxes }: { boxes: Box[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useGSAP(
    () => {
      if (prefersReduced()) return;
      gsap.from(".region", { scale: 0.96, opacity: 0, stagger: 0.06, duration: 0.5 });
    },
    { scope: ref, dependencies: [boxes] },
  );
  return (
    <div ref={ref} className="pointer-events-none absolute inset-0">
      {boxes.map((b, i) => (
        <RegionBox key={i} box={b} tone="olive" label={b.label} />
      ))}
    </div>
  );
}
