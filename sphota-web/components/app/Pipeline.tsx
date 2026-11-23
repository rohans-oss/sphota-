"use client";
import { Check } from "@phosphor-icons/react";
import { useJob } from "@/lib/store";

const ROWS = [
  ["detect", "Detect text and faces"], ["recommend", "Recommend mode"], ["encode", "Encode with VAE"],
  ["entropy", "Entropy-code with GMM"], ["pack", "Pack the seed"],
];

// waiting (dashed ring) -> running (amber pulse) -> done (olive tick + time), fed by server events (guide F16).
export function Pipeline() {
  const stages = useJob((s) => s.stages);
  return (
    <ul aria-label="Pipeline">
      {ROWS.map(([k, label]) => {
        const s = stages[k]?.status ?? "waiting";
        return (
          <li key={k} className={"flex items-center gap-3 border-t border-gold-200/10 py-2.5 text-sm " + (s === "running" ? "text-amber-400" : s === "waiting" ? "opacity-50" : "")}>
            <span className={"grid size-5 place-items-center rounded-full " + (s === "done" ? "bg-olive text-mocha-950" : s === "running" ? "animate-[pulse-ring_1.6s_ease-out_infinite] border-2 border-amber-400" : "border border-dashed border-gold-400")}>
              {s === "done" && <Check size={12} weight="bold" />}
            </span>
            {label}
            <span className="ml-auto font-mono text-xs text-gold-400">{s === "done" ? ((stages[k].ms ?? 0) / 1000).toFixed(2) + " s" : s}</span>
          </li>
        );
      })}
    </ul>
  );
}
