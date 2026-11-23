// One fixed film-grain overlay for the whole site (guide F3).
const NOISE =
  "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'>" +
  "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/>" +
  "<feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .9 0'/></filter>" +
  "<rect width='100%' height='100%' filter='url(%23n)'/></svg>";

export function Grain() {
  return (
    <div
      aria-hidden
      className="grain no-print pointer-events-none fixed inset-[-10%] z-[60] opacity-[.28] mix-blend-multiply"
      style={{ backgroundImage: `url("${NOISE}")` }}
    />
  );
}
