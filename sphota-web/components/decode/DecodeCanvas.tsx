"use client";
import { useEffect, useRef } from "react";
import { gsap } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";
import { onceFired } from "@/lib/events";

type Props = { src: string; playOn?: string; delay?: number; duration?: number; progress?: number; label?: string };

const hash = (x: number, y: number, t: number) => {
  const s = Math.sin(x * 127.1 + y * 311.7 + t * 74.7) * 43758.5453;
  return s - Math.floor(s);
};

// The photo arrives as a coarse, noisy seed grid that refines into the real image (guide F9).
export function DecodeCanvas({ src, playOn, delay = 0, duration = 2.8, progress, label = "Photo decoded from a 1.9 KB seed" }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const draw = useRef<(p: number) => void>(() => {});

  useEffect(() => {
    const c = ref.current!, ctx = c.getContext("2d")!;
    const img = new Image();
    img.src = src;
    const small = document.createElement("canvas"), noise = document.createElement("canvas");
    let cancelled = false, tween: gsap.core.Tween | undefined, cleanupListener: (() => void) | undefined;
    img.decode().then(() => {
      if (cancelled) return;
      const r = c.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
      c.width = Math.max(2, r.width * dpr);
      c.height = Math.max(2, r.height * dpr);
      draw.current = (p) => {
        const W = c.width, H = c.height;
        const cells = Math.round(10 + 590 * Math.pow(p, 2.2)), rows = Math.max(1, Math.round((cells * H) / W));
        small.width = cells; small.height = rows;
        small.getContext("2d")!.drawImage(img, 0, 0, cells, rows);
        ctx.globalCompositeOperation = "source-over"; ctx.imageSmoothingEnabled = false; ctx.globalAlpha = 1;
        ctx.drawImage(small, 0, 0, W, H); // the seed grid
        const a = Math.min(1, Math.max(0, (p - 0.82) / 0.18));
        if (a > 0) {
          ctx.globalAlpha = a * a * (3 - 2 * a); ctx.imageSmoothingEnabled = true;
          ctx.drawImage(img, 0, 0, W, H); ctx.globalAlpha = 1;
        }
        const amt = 0.75 * (1 - p); // diffusion noise fades out
        if (amt > 0.01) {
          noise.width = cells; noise.height = rows;
          const nx = noise.getContext("2d")!, id = nx.createImageData(cells, rows), t = Math.floor(p * 40);
          for (let y = 0; y < rows; y++)
            for (let x = 0; x < cells; x++) {
              const v = hash(x, y, t) * 255, i = (y * cells + x) * 4;
              id.data[i] = id.data[i + 1] = id.data[i + 2] = v;
              id.data[i + 3] = amt * 255;
            }
          nx.putImageData(id, 0, 0);
          ctx.globalCompositeOperation = "overlay"; ctx.imageSmoothingEnabled = false;
          ctx.drawImage(noise, 0, 0, W, H);
          ctx.globalCompositeOperation = "source-over";
        }
      };
      if (progress !== undefined) { draw.current(progress); return; } // driven by scroll (F11)
      if (prefersReduced()) { draw.current(1); return; }
      draw.current(0);
      const state = { p: 0 };
      const run = () => { tween = gsap.to(state, { p: 1, duration, delay, ease: "power2.inOut", onUpdate: () => draw.current(state.p) }); };
      if (playOn) cleanupListener = onceFired(playOn, run);
      else run();
    }).catch(() => {});
    return () => { cancelled = true; tween?.kill(); cleanupListener?.(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  useEffect(() => { if (progress !== undefined) draw.current(progress); }, [progress]);
  return <canvas ref={ref} className="size-full" role="img" aria-label={label} />;
}
