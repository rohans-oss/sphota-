"use client";
import { useRef, useState } from "react";
import { gsap, useGSAP } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";
import { fire } from "@/lib/events";
import { Sun } from "@/components/retro/Sun";

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
const decode = (src: string) => {
  const i = new Image();
  i.src = src;
  return i.decode().catch(() => {});
};
export const playHero = () => fire("hero:play");

// First visit per session only; real progress = fonts + hero image, 1.2 s minimum, 3 s maximum (guide F7).
export function Loader() {
  const root = useRef<HTMLDivElement>(null);
  const [gone, setGone] = useState(false);

  useGSAP(
    () => {
      if (prefersReduced() || sessionStorage.getItem("seen")) {
        setGone(true);
        playHero();
        return;
      }
      document.documentElement.style.overflow = "hidden";
      const state = { p: 0 };
      const count = root.current!.querySelector(".count")!;
      const render = () => { count.textContent = String(Math.round(state.p)).padStart(3, "0"); };
      gsap.from(".mark", { yPercent: 110, duration: 0.8 });
      gsap.fromTo(".loader-sun", { yPercent: 60 }, { yPercent: 0, duration: 2.2, ease: "power2.out" });
      const gather = gsap.to(state, { p: 90, duration: 2, ease: "power2.out", onUpdate: render });
      const ready = Promise.all([document.fonts.ready, decode("/hero.webp"), wait(1200)]);
      Promise.race([ready, wait(3000)]).then(() => {
        gather.kill();
        gsap
          .timeline({
            onComplete: () => {
              sessionStorage.setItem("seen", "1");
              document.documentElement.style.overflow = "";
              setGone(true);
            },
          })
          .to(state, { p: 100, duration: 0.3, onUpdate: render })
          .to(count, { duration: 0.4, scrambleText: { text: "1.9 KB", chars: "0123456789" } })
          .to(root.current, { clipPath: "inset(0 0 100% 0)", duration: 1, ease: "wipe" }, "+=0.1")
          .call(playHero, [], "-=0.3");
      });
      return () => { document.documentElement.style.overflow = ""; };
    },
    { scope: root },
  );

  if (gone) return null;
  return (
    <div
      ref={root} role="status" aria-label="Loading Sphoṭa"
      className="loader-root fixed inset-0 z-[70] grid place-items-center bg-mocha-700" style={{ clipPath: "inset(0 0 0% 0)" }}
    >
      <div className="flex flex-col items-center">
        <div className="relative h-[150px] w-[min(600px,90vw)] overflow-hidden border-b-[1.5px] border-amber-600">
          <Sun size={300} id="lsun" className="loader-sun absolute bottom-0 left-1/2 w-[300px] -translate-x-1/2" />
        </div>
        <div className="mt-10 overflow-hidden">
          <p className="mark font-logo text-[64px] text-gold-100">Sphoṭa</p>
        </div>
        <p className="lab mt-4 !tracking-[.3em] text-gold-200">Receiving transmission</p>
        <span className="count mt-5 font-mono text-[44px] font-medium tracking-[.12em] text-amber-400">000</span>
      </div>
    </div>
  );
}
