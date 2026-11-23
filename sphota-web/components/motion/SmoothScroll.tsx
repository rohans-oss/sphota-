"use client";
import Lenis from "lenis";
import "lenis/dist/lenis.css";
import { useEffect } from "react";
import { gsap, ScrollTrigger } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";

// Lenis smooth scroll synced with GSAP's ticker (guide 6.4). Landing page only; native scroll under reduced motion.
export function SmoothScroll({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    if (prefersReduced()) return;
    const lenis = new Lenis({ autoRaf: false, lerp: 0.09, wheelMultiplier: 0.9 });
    const tick = (t: number) => lenis.raf(t * 1000);
    gsap.ticker.add(tick);
    gsap.ticker.lagSmoothing(0);
    lenis.on("scroll", ScrollTrigger.update);
    return () => {
      gsap.ticker.remove(tick);
      lenis.destroy();
    };
  }, []);
  return <>{children}</>;
}
