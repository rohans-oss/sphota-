// Four parallel bands; the hero draws them with DrawSVG (guide F5).
const BANDS = ["#F5E7C6", "#E9B66A", "#E07A2E", "#C9561E"];
const D = "M-40 770 C 300 770, 560 700, 760 560 S 1160 250, 1500 200";

export function Ribbon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice" className={className} aria-hidden>
      {BANDS.map((c, i) => (
        <path key={c} className="ribbon-band" d={D} transform={`translate(0 ${i * 22})`} fill="none" stroke={c} strokeWidth={20} opacity={0.92} />
      ))}
    </svg>
  );
}
