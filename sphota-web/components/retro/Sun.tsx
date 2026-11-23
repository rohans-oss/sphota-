// A half disc with a cream-to-amber gradient, cut by bands (guide F5).
type SunProps = { size?: number; bands?: number; bandColor?: string; id?: string; className?: string };

export function Sun({ size = 600, bands = 7, bandColor = "var(--color-mocha-700)", id = "sun", className }: SunProps) {
  const r = size / 2;
  return (
    <svg viewBox={`0 0 ${size} ${r}`} className={className} aria-hidden>
      <defs>
        <linearGradient id={id + "-g"} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#F5E7C6" />
          <stop offset=".45" stopColor="#E9B66A" />
          <stop offset="1" stopColor="#C9561E" />
        </linearGradient>
      </defs>
      <circle className="sun-disc" cx={r} cy={r} r={r} fill={`url(#${id}-g)`} />
      {Array.from({ length: bands }, (_, i) => (
        <rect key={i} className="sun-band" x="0" width={size} y={r * 0.45 + i * r * 0.08} height={2 + i * 2.2} fill={bandColor} />
      ))}
    </svg>
  );
}
