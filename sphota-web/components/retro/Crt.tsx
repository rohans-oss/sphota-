import clsx from "clsx";
import type { CSSProperties, ReactNode } from "react";

type CrtProps = { children: ReactNode; className?: string; accent?: boolean; tilt?: number; style?: CSSProperties };

// Rounded CRT screen with scanlines and an inner vignette (guide F5).
export function Crt({ children, className, accent, tilt = 0, style }: CrtProps) {
  return (
    <div
      className={clsx("crt rounded-[30px] p-3.5 shadow-[0_26px_50px_-20px_rgba(30,12,5,.7)]", accent ? "bg-amber-400" : "bg-gold-200", className)}
      style={{ transform: tilt ? `rotate(${tilt}deg)` : undefined, ...style }}
    >
      <div className="crt-screen scanlines relative size-full overflow-hidden rounded-[20px] bg-[#2a1a12]">{children}</div>
    </div>
  );
}
