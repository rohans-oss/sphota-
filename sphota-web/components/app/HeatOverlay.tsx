"use client";
import { useEffect, useRef } from "react";

// Grayscale uncertainty PNG -> transparent / amber / rust overlay (guide F18).
const STOPS: [number, number, number, number][] = [[224, 122, 46, 0], [224, 122, 46, 0.55], [210, 69, 46, 0.85]];
function ramp(t: number) {
  const s = t < 0.5 ? 0 : 1, k = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  return STOPS[s].map((v, i) => v + (STOPS[s + 1][i] - v) * k);
}

export function HeatOverlay({ src, visible }: { src: string; visible: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = src;
    img.onload = () => {
      const c = ref.current;
      if (!c) return;
      c.width = img.width; c.height = img.height;
      const x = c.getContext("2d")!;
      x.drawImage(img, 0, 0);
      const d = x.getImageData(0, 0, c.width, c.height);
      for (let i = 0; i < d.data.length; i += 4) {
        const [r, g, b, a] = ramp(d.data[i] / 255);
        d.data[i] = r; d.data[i + 1] = g; d.data[i + 2] = b; d.data[i + 3] = a * 255;
      }
      x.putImageData(d, 0, 0);
    };
  }, [src]);
  return <canvas ref={ref} aria-hidden className={"pointer-events-none absolute inset-0 size-full transition-opacity duration-500 " + (visible ? "opacity-90" : "opacity-0")} />;
}
