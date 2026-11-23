"use client";
import Link from "next/link";
import { useRef } from "react";
import { ArrowUpRight } from "@phosphor-icons/react";
import { gsap, useGSAP } from "@/lib/motion";
import { Sun } from "@/components/retro/Sun";
import { Horizon } from "@/components/retro/Horizon";

// A second banded sun sets behind the horizon grid as the band scrolls in (guide F13).
export function Cta() {
  const ref = useRef<HTMLElement>(null);
  useGSAP(
    () => {
      gsap.matchMedia().add("(prefers-reduced-motion: no-preference)", () => {
        gsap.fromTo(".cta-sun", { y: -30 }, { y: 0, ease: "none", scrollTrigger: { trigger: ref.current, start: "top bottom", end: "bottom 60%", scrub: true } });
      });
    },
    { scope: ref },
  );
  return (
    <section id="cta" ref={ref} className="relative min-h-[340px] overflow-hidden bg-mocha-900 px-6 py-16 md:px-14">
      <Sun size={400} id="csun" className="cta-sun absolute bottom-[70px] right-[-40px] w-[320px] md:right-[8%] md:w-[400px]" />
      <Horizon className="!h-[24%]" />
      <div className="relative z-10 max-w-[560px] pb-24">
        <p className="font-deva text-[40px] leading-none text-amber-400">स्फोट</p>
        <h2 className="h2-serif mt-3">Send your first <em>seed.</em></h2>
        <p className="mt-4 text-gold-200">Drop a photo, fold it into a seed, and watch it burst back with every word intact.</p>
        <div className="mt-6 flex flex-wrap gap-4">
          <Link href="/compress" className="btn-primary">Send a photo <ArrowUpRight size={18} weight="bold" /></Link>
          <a href="#faq" className="btn-ghost">Read the paper</a>
        </div>
      </div>
    </section>
  );
}
