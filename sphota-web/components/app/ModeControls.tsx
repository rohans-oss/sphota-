"use client";
import { Segmented } from "@/components/ui/Segmented";
import { Slider } from "@/components/ui/Slider";
import type { Mode } from "@/lib/types";

const MODES = [["exact", "Exact"], ["balanced", "Balanced"], ["small", "Small"], ["tiny", "Tiny"]] as const;
export const NETWORKS = ["2G", "3G", "4G", "WiFi"] as const;

export function ModeControls({
  mode, setMode, network, setNetwork, targetKb, setTargetKb, recommended,
}: {
  mode: Mode; setMode: (m: Mode) => void; network: string; setNetwork: (n: string) => void;
  targetKb: number; setTargetKb: (n: number) => void; recommended?: Mode;
}) {
  return (
    <div className="space-y-6">
      <div>
        <p className="lab mb-2">Mode{recommended ? <span className="ml-2 normal-case tracking-normal text-amber-400">Recommended: {recommended[0].toUpperCase() + recommended.slice(1)}</span> : null}</p>
        <Segmented value={mode} onChange={setMode} options={MODES} label="Mode" />
      </div>
      <div>
        <p className="lab mb-2">Network</p>
        <div className="flex flex-wrap gap-2">
          {NETWORKS.map((n) => (
            <button key={n} type="button" className="chip min-h-11 px-4" data-active={network === n} aria-pressed={network === n} onClick={() => setNetwork(n)}>{n}</button>
          ))}
        </div>
      </div>
      <Slider value={targetKb} onChange={setTargetKb} min={0.5} max={8} step={0.1} label="Target size" />
    </div>
  );
}
