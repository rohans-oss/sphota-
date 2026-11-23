"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { List, X } from "@phosphor-icons/react";
import { gsap } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";

// Full-height mocha menu under 768 px. Focus is trapped inside while open; Escape closes (guide F6).
export function MobileMenu({ links }: { links: [string, string][] }) {
  const [open, setOpen] = useState(false);
  const panel = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const el = panel.current!;
    document.documentElement.style.overflow = "hidden";
    if (!prefersReduced()) gsap.from(el.querySelectorAll("a"), { y: 24, opacity: 0, stagger: 0.06 });
    const focusables = () => Array.from(el.querySelectorAll<HTMLElement>("a,button"));
    focusables()[0]?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
      if (e.key === "Tab") {
        const f = [...focusables(), button.current!];
        const i = f.indexOf(document.activeElement as HTMLElement);
        const next = e.shiftKey ? (i <= 0 ? f.length - 1 : i - 1) : (i + 1) % f.length;
        e.preventDefault();
        f[next].focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.documentElement.style.overflow = "";
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="ml-auto md:hidden">
      <button
        ref={button} type="button" aria-expanded={open} aria-label={open ? "Close menu" : "Open menu"} onClick={() => setOpen(!open)}
        className={"relative z-[90] grid size-11 place-items-center rounded-full transition-colors " + (open ? "bg-gold-200 text-mocha-900" : "border border-gold-200/40 text-gold-100")}
      >
        {open ? <X size={20} /> : <List size={20} />}
      </button>
      {open && (
        <div ref={panel} className="fixed inset-0 z-[80] flex flex-col justify-center gap-5 bg-mocha-900 px-8">
          {links.map(([href, label]) => (
            <Link key={href + label} href={href} onClick={() => setOpen(false)} className="font-display text-[50px] leading-none text-gold-100" style={{ fontVariationSettings: '"SOFT" 100' }}>
              {label}
            </Link>
          ))}
          <div aria-hidden className="horizon !h-[22%]"><div className="horizon-plane" /></div>
        </div>
      )}
    </div>
  );
}
