import { Sun } from "@/components/retro/Sun";
import { Horizon } from "@/components/retro/Horizon";
import { Ribbon } from "@/components/retro/Ribbon";
import { Crt } from "@/components/retro/Crt";

export const metadata = { title: "Tokens · Sphoṭa", robots: { index: false } };

const SWATCHES: [string, string][] = [
  ["mocha-950", "#3E271C"], ["mocha-900", "#4A2F22"], ["mocha-700", "#5A3A2B"], ["mocha-600", "#6E4A37"], ["mocha-500", "#86593F"],
  ["gold-100", "#F5E7C6"], ["gold-200", "#E9D4A4"], ["gold-400", "#BFA57A"], ["amber-400", "#E07A2E"], ["amber-600", "#C9561E"],
  ["sun", "#E9B66A"], ["olive", "#A9B970"], ["rust", "#D2452E"],
];

// Hidden visual test sheet for the design system (guide F2 and F5).
export default function Tokens() {
  return (
    <main id="main" className="space-y-16 px-6 py-16 md:px-14">
      <section>
        <p className="eyebrow">Tokens · colour</p>
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
          {SWATCHES.map(([n, h]) => (
            <div key={n} className="rounded-2xl border border-gold-200/20 p-3">
              <div className="h-16 rounded-lg" style={{ background: h }} />
              <p className="mt-2 font-mono text-xs">{n}</p>
              <p className="font-mono text-xs text-gold-400">{h}</p>
            </div>
          ))}
        </div>
      </section>
      <section className="space-y-4">
        <p className="eyebrow">Tokens · type</p>
        <p className="display">The whole <em>picture.</em></p>
        <p className="h2-serif">Heading two, <em>serif</em></p>
        <p className="font-logo text-[30px]">Sphoṭa · Shrikhand</p>
        <p className="font-deva text-[40px] text-amber-400">स्फोट</p>
        <p className="lab">lab · LPIPS 0.21 · 1.9 KB</p>
        <p className="kpi text-[64px]">21.7×</p>
        <p className="max-w-[640px] text-[19px] leading-[29px] text-gold-200">Space Grotesk body text. Sphoṭa sends a tiny seed instead of a heavy photo.</p>
        <p className="font-mono">Geist Mono · 000 → 1.9 KB</p>
      </section>
      <section className="space-y-4">
        <p className="eyebrow">Tokens · controls</p>
        <div className="flex flex-wrap items-center gap-4">
          <button className="btn-primary">Send a photo</button>
          <button className="btn-action">Compress to seed</button>
          <button className="btn-ghost">Ghost</button>
          <span className="chip">chip</span>
          <span className="chip" data-active="true">active</span>
        </div>
        <div className="panel max-w-[420px] p-6">Panel · rounded-3xl bg-mocha-600</div>
      </section>
      <section>
        <p className="eyebrow">Artwork · Sun, Horizon, Ribbon, Crt</p>
        <div className="relative mt-6 h-[520px] overflow-hidden rounded-3xl border border-gold-200/20">
          <Sun size={420} id="tsun" className="absolute left-10 top-16 w-[420px]" />
          <Horizon />
          <Ribbon className="absolute inset-0 size-full" />
          <Crt tilt={3} className="absolute right-10 top-24 h-[240px] w-[320px]">
            <div className="grid size-full place-items-center font-mono text-gold-200">empty CRT</div>
          </Crt>
        </div>
      </section>
    </main>
  );
}
