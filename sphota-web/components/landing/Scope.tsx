"use client";
import { DecodeCanvas } from "@/components/decode/DecodeCanvas";

// The CRT picture for each stage: SVG with an amber phosphor look (guide F11). Everything is driven by props so
// scrubbing is perfectly reversible.
function Grid() {
  return (
    <g stroke="#E07A2E" strokeOpacity=".16" strokeWidth="1">
      {Array.from({ length: 13 }, (_, i) => <line key={"v" + i} x1={i * 56} y1="0" x2={i * 56} y2="442" />)}
      {Array.from({ length: 9 }, (_, i) => <line key={"h" + i} x1="0" y1={i * 55.25} x2="672" y2={i * 55.25} />)}
    </g>
  );
}

function LatentGrid({ fill }: { fill: number }) {
  const cols = 12, rows = 6;
  return (
    <g>
      {Array.from({ length: cols * rows }, (_, k) => {
        const c = k % cols, r = Math.floor(k / cols);
        const t = Math.min(1, Math.max(0, fill * 1.6 - c / cols));
        const shade = ["#F5E7C6", "#E9B66A", "#E07A2E", "#C9561E"][(c * 7 + r * 3) % 4];
        return <rect key={k} x={36 + c * 50} y={52 + r * 56} width="42" height="46" rx="5" fill={shade} opacity={0.15 + 0.85 * t} />;
      })}
    </g>
  );
}

function Gmm({ draw }: { draw: number }) {
  const comps = [[220, 50, 0.8, "#E9D4A4"], [330, 30, 1.25, "#E07A2E"], [450, 62, 0.55, "#C9561E"]] as const;
  return (
    <g>
      <line x1="30" y1="360" x2="640" y2="360" stroke="#BFA57A" strokeOpacity=".5" />
      {comps.map(([m, s, a, c]) => {
        let d = "M30 360";
        for (let x = 30; x <= 640; x += 3) d += " L" + x + " " + (360 - a * 240 * Math.exp(-((x - m) ** 2) / (2 * s * s)));
        return <path key={c} d={d} fill="none" stroke={c} strokeWidth={3} pathLength={1} strokeDasharray="1" strokeDashoffset={1 - draw} />;
      })}
    </g>
  );
}

function BitStream({ travel }: { travel: number }) {
  return (
    <g>
      {Array.from({ length: 30 }, (_, i) => {
        const x = ((i * 22 + travel * 700) % 700) - 20;
        const on = (i * 5) % 3 !== 0;
        return <rect key={i} x={x} y={190 + ((i * 17) % 3) * 24} width="14" height="14" fill={on ? "#E07A2E" : "#E9D4A4"} opacity={on ? 0.95 : 0.6} />;
      })}
      <text x="336" y="140" textAnchor="middle" fontFamily="var(--font-geist-mono)" fontSize="30" fill="#F5E7C6">1.9 KB</text>
    </g>
  );
}

export function Scope({ stage, progress }: { stage: number; progress: number }) {
  return (
    <div className="relative size-full">
      <svg viewBox="0 0 672 442" className="absolute inset-0 size-full" aria-hidden>
        <Grid />
        {stage === 0 && <LatentGrid fill={progress} />}
        {stage === 1 && <Gmm draw={progress} />}
        {stage === 2 && <BitStream travel={progress} />}
      </svg>
      <div className={"absolute inset-0 transition-opacity duration-300 " + (stage === 3 ? "opacity-100" : "opacity-0")}>
        <DecodeCanvas src="/hero.webp" progress={stage === 3 ? progress : 0} label="Photo decoding from the seed" />
      </div>
    </div>
  );
}
