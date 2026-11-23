"use client";
import Link from "next/link";
import { useRef } from "react";
import { ArrowUpRight } from "@phosphor-icons/react";
import { gsap, SplitText, useGSAP, crtOn } from "@/lib/motion";
import { onceFired } from "@/lib/events";
import { Sun } from "@/components/retro/Sun";
import { Horizon } from "@/components/retro/Horizon";
import { Ribbon } from "@/components/retro/Ribbon";
import { Crt } from "@/components/retro/Crt";
import { DecodeCanvas } from "@/components/decode/DecodeCanvas";
import { DecodeNumber } from "@/components/motion/DecodeNumber";

export function Hero() {
  const hero = useRef<HTMLElement>(null);

  // entrance (waits for the loader) and scroll parallax (guide F8)
  useGSAP(
    () => {
      const mm = gsap.matchMedia();
      mm.add("(prefers-reduced-motion: no-preference)", () => {
        const split = SplitText.create(".hero-title", { type: "lines", mask: "lines", autoSplit: true });
        gsap.set(".ribbon-band", { drawSVG: "0%" });
        const tl = gsap
          .timeline({ paused: true })
          .from(".nav-shell > *", { y: -20, opacity: 0, stagger: 0.06, duration: 0.6 }, 0)
          .from(split.lines, { yPercent: 115, rotate: 2.5, duration: 1.1, stagger: 0.09 }, 0)
          .from(".hero-fade", { y: 24, opacity: 0, stagger: 0.08, duration: 0.8 }, 0.5)
          .to(".ribbon-band", { drawSVG: "100%", duration: 1.4, stagger: 0.08, ease: "power2.inOut" }, 0)
          .from([".hero-sun", ".horizon"], { y: 60, opacity: 0, duration: 1.4 }, 0)
          .add(crtOn(".hero-crt"), 0.9)
          .from(".hero-stat", { y: 20, opacity: 0, duration: 0.6 }, 1.2);
        const off = onceFired("hero:play", () => tl.play());
        gsap
          .timeline({ scrollTrigger: { trigger: hero.current, start: "top top", end: "bottom top", scrub: true } })
          .to(".hero-sun", { y: 80, ease: "none" }, 0)
          .to(".hero-crt", { y: -60, ease: "none" }, 0)
          .to(".ribbon-band", { x: 120, opacity: 0.4, ease: "none" }, 0);
        return () => { off(); split.revert(); };
      });
    },
    { scope: hero },
  );

  return (
    <section ref={hero} className="hero relative min-h-[100svh] overflow-hidden md:h-[100svh] md:min-h-[760px]">
      <div className="absolute inset-x-0 top-0 h-[360px] md:inset-0 md:h-auto">
        <Sun size={600} id="hsun" className="hero-sun absolute left-1/2 top-[70px] w-[360px] -translate-x-1/2 md:left-auto md:right-[-60px] md:top-[120px] md:w-[600px] md:translate-x-0" />
        <Horizon className="!h-[40%] md:!h-[38%]" />
        <Ribbon className="pointer-events-none absolute inset-0 hidden size-full md:block" />
      </div>
      <div className="relative z-10 px-6 pt-[400px] pb-16 md:max-w-[700px] md:pb-0 md:pl-14 md:pr-0 md:pt-[150px]">
        <p className="eyebrow hero-fade">Transmission 01 · 1.9 KB</p>
        <h1 className="display hero-title mt-6">
          The whole<br />picture,<br /><em>in a seed.</em>
        </h1>
        <p className="hero-fade mt-7 max-w-[450px] text-[19px] leading-[29px] text-gold-200">
          Sphoṭa sends a tiny seed instead of a heavy photo. A diffusion model grows it back on the other side,
          and the doses, names and faces arrive exactly as they left.
        </p>
        <div className="hero-fade mt-8 flex flex-wrap items-center gap-6">
          <Link href="/compress" className="btn-primary max-md:w-full max-md:justify-center">
            Send a photo <ArrowUpRight size={18} weight="bold" />
          </Link>
          <a href="#how" className="text-gold-100">How it works ↓</a>
        </div>
      </div>
      <Crt tilt={3} className="hero-crt absolute right-6 top-[150px] h-[150px] w-[200px] md:right-[110px] md:top-[300px] md:h-[300px] md:w-[400px]">
        <DecodeCanvas src="/hero.webp" playOn="hero:play" delay={1.1} />
      </Crt>
      <div className="hero-stat absolute bottom-[120px] right-[56px] hidden text-right md:block">
        <DecodeNumber value="21.7×" className="kpi text-[64px]" />
        <p className="lab mt-1">Smaller than the JPEG</p>
        <p className="lab mt-1 !normal-case !tracking-normal text-gold-400/80">demo value</p>
      </div>
    </section>
  );
}
