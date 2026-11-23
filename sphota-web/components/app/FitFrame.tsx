"use client";
import { useState } from "react";

// Shows an image at its own aspect ratio inside a CRT screen, so fractional boxes and overlays always line up
// with the picture (guide F12/F18). Children are overlays positioned in percent.
export function FitFrame({
  src, alt, screenRatio, children, imgClassName = "",
}: { src: string; alt: string; screenRatio: number; children?: React.ReactNode; imgClassName?: string }) {
  const [ar, setAr] = useState(screenRatio);
  const wide = ar > screenRatio;
  return (
    <div className="grid size-full place-items-center">
      <div className="relative" style={{ aspectRatio: ar, width: wide ? "100%" : undefined, height: wide ? undefined : "100%" }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={alt} className={"size-full object-fill " + imgClassName} onLoad={(e) => setAr(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)} />
        {children}
      </div>
    </div>
  );
}
