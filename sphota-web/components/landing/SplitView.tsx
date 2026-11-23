"use client";
import { useEffect, useRef, useState } from "react";

// Fullscreen before/after with a range-input divider: keyboard accessible, Escape closes (guide F12).
export function SplitView({ left, right, onClose }: { left: string; right: string; onClose: () => void }) {
  const [pos, setPos] = useState(50);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    input.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <dialog open aria-label="Fullscreen comparison" className="fixed inset-0 z-[80] grid size-full max-h-none max-w-none place-items-center bg-mocha-950/95 p-10 text-gold-100">
      <div className="relative aspect-[4/3] h-[min(80vh,70vw)] overflow-hidden rounded-3xl">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={left} alt="Comparison decode" className="absolute inset-0 size-full object-cover" />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={right} alt="Sphoṭa decode" className="absolute inset-0 size-full object-cover" style={{ clipPath: `inset(0 0 0 ${pos}%)` }} />
        <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-gold-100" style={{ left: pos + "%" }} />
        <span className="lab absolute left-4 top-4 rounded bg-mocha-950/70 px-2 py-1">Other decoder</span>
        <span className="lab absolute right-4 top-4 rounded bg-mocha-950/70 px-2 py-1">Sphoṭa</span>
        <input
          ref={input} type="range" min={0} max={100} value={pos} onChange={(e) => setPos(+e.target.value)}
          aria-label="Move the divider" className="absolute inset-0 size-full cursor-ew-resize opacity-0"
        />
      </div>
      <button onClick={onClose} className="btn-ghost mt-6">Close</button>
    </dialog>
  );
}
