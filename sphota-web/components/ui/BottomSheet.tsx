"use client";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";

export function useIsDesktop() {
  const [d, setD] = useState(true);
  useEffect(() => {
    const m = window.matchMedia("(min-width: 1024px)");
    const f = () => setD(m.matches);
    f();
    m.addEventListener("change", f);
    return () => m.removeEventListener("change", f);
  }, []);
  return d;
}

// Inline on desktop; a draggable bottom sheet on phones (guide F20). Drag down to close.
export function Sheet({ title, children, openLabel }: { title: string; children: React.ReactNode; openLabel?: string }) {
  const desktop = useIsDesktop();
  const [open, setOpen] = useState(false);
  if (desktop) return <>{children}</>;
  return (
    <>
      <button className="btn-ghost w-full justify-center" onClick={() => setOpen(true)}>{openLabel ?? title}</button>
      <AnimatePresence>
        {open && (
          <>
            <motion.div className="fixed inset-0 z-[70] bg-mocha-950/70" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setOpen(false)} />
            <motion.div
              role="dialog" aria-label={title} drag="y" dragConstraints={{ top: 0, bottom: 0 }} dragElastic={{ top: 0, bottom: 0.6 }}
              onDragEnd={(_, i) => i.offset.y > 120 && setOpen(false)}
              className="fixed inset-x-0 bottom-0 z-[75] max-h-[80svh] overflow-y-auto rounded-t-3xl bg-mocha-900 p-6"
              initial={{ y: "100%" }} animate={{ y: 0 }} exit={{ y: "100%" }} transition={{ type: "spring", bounce: 0.1, duration: 0.5 }}
            >
              <div className="mx-auto mb-4 h-1.5 w-12 rounded-full bg-gold-400/50" />
              <h3 className="mb-4 font-display text-2xl">{title}</h3>
              {children}
              <button className="btn-ghost mt-6 w-full justify-center" onClick={() => setOpen(false)}>Close</button>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
