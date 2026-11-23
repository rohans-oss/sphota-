"use client";
import { useRef } from "react";
import { gsap, useGSAP } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";

// Numbers scramble into their value once (guide 6.6). The real value is already in the HTML.
export function DecodeNumber({ value, className }: { value: string; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  useGSAP(
    () => {
      if (prefersReduced() || !ref.current) return;
      gsap.from(ref.current, {
        duration: 1.1,
        scrambleText: { text: value, chars: "0123456789.", revealDelay: 0.3 },
        scrollTrigger: { trigger: ref.current, start: "top 88%", once: true },
      });
    },
    { dependencies: [value] },
  );
  return (
    <span ref={ref} className={className}>
      {value}
    </span>
  );
}
