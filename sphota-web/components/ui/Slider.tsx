"use client";

export function Slider({
  value, onChange, min, max, step = 0.1, label, unit = "KB",
}: { value: number; onChange: (v: number) => void; min: number; max: number; step?: number; label: string; unit?: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label className="lab" htmlFor="slider-input">{label}</label>
        <span className="font-mono text-sm text-gold-100">{value.toFixed(1)} {unit}</span>
      </div>
      <input id="slider-input" className="slider mt-2" type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}
