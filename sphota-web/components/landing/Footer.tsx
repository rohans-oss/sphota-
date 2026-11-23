"use client";
import Link from "next/link";
import { useSyncExternalStore } from "react";
import { Toggle } from "@/components/ui/Toggle";

// Footer with the Reduce motion toggle: stored, applied before paint by an inline script in the layout (guide F13).
export function Footer() {
  const reduced = useSyncExternalStore(
    (cb) => { window.addEventListener("storage", cb); return () => window.removeEventListener("storage", cb); },
    () => localStorage.getItem("motion") === "reduced",
    () => false,
  );
  function toggle(next: boolean) {
    localStorage.setItem("motion", next ? "reduced" : "full");
    document.documentElement.dataset.motion = next ? "reduced" : "full";
    location.reload(); // simplest way to rebuild every animation
  }
  return (
    <footer className="bg-mocha-950 px-6 py-12 md:px-14">
      <div className="flex flex-wrap items-start justify-between gap-8">
        <div>
          <p className="font-logo text-[30px] text-gold-100">Sphoṭa</p>
          <p className="mt-2 text-gold-400">From the seed, only the truth.</p>
        </div>
        <nav aria-label="Footer" className="flex flex-wrap gap-x-8 gap-y-3 text-gold-200">
          {["Research", "Paper", "GitHub", "Privacy", "Terms"].map((l) => (
            <Link key={l} href="#faq" className="hover:text-gold-100">{l}</Link>
          ))}
        </nav>
        <Toggle checked={reduced} onChange={toggle} label="Reduce motion" />
      </div>
      <p className="lab mt-10">Sphoṭa · a faithful generative image codec · demo values shown until evaluation results replace them</p>
    </footer>
  );
}
