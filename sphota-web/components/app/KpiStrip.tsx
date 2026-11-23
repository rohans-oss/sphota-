"use client";
import { DecodeNumber } from "@/components/motion/DecodeNumber";
import type { CompressResult } from "@/lib/types";

export function KpiStrip({ result }: { result?: CompressResult }) {
  const enc = result ? Object.values(result.timings).reduce((a, b) => a + b, 0) : 0;
  const items: [string, string, boolean][] = [
    ["Seed size", result ? (result.bytes / 1000).toFixed(1) + " KB" : "—", true],
    ["Smaller by", result ? Math.round(result.original_bytes / result.bytes) + "×" : "—", false],
    ["Bits per pixel", result ? result.bpp.toFixed(3) : "—", false],
    ["Protected", result ? result.boxes.length + " regions" : "—", false],
    ["Encode time", result ? enc.toFixed(1) + " s" : "—", false],
  ];
  return (
    <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-3 lg:grid-cols-5">
      {items.map(([k, v, hero]) => (
        <div key={k} className="border-t border-gold-200/20 pt-3">
          <dt className="lab">{k}</dt>
          <dd className={"kpi mt-1 " + (hero ? "text-[40px]" : "!text-gold-100 text-[28px]")}>
            {result ? <DecodeNumber key={result.job_id + k} value={v} /> : <span className="font-mono text-[24px] opacity-40">--</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}
