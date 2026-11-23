"use client";
import { GridFour, Pill, TextT, Warning } from "@phosphor-icons/react";
import type { Flag } from "@/lib/types";

const ICONS = { pill: Pill, texture: GridFour, text: TextT } as const;

// Hovering a row highlights its box on the screen, and the reverse (guide F18).
export function FlagList({ flags, hover, setHover }: { flags: Flag[]; hover?: string; setHover: (id?: string) => void }) {
  return (
    <ul aria-label="Flagged regions" className="space-y-2">
      {flags.map((f) => {
        const Icon = ICONS[f.icon as keyof typeof ICONS] ?? Warning;
        return (
          <li
            key={f.id} onMouseEnter={() => setHover(f.id)} onMouseLeave={() => setHover(undefined)}
            onFocus={() => setHover(f.id)} onBlur={() => setHover(undefined)} tabIndex={0}
            className={"flex items-start gap-3 rounded-2xl px-4 py-3 transition-colors " + (hover === f.id ? "bg-mocha-900 outline outline-1 outline-rust/60" : "")}
          >
            <Icon size={20} className="mt-0.5 shrink-0 text-amber-400" />
            <div className="min-w-0 flex-1">
              <p className="font-medium">{f.name}</p>
              <p className="text-sm text-gold-400">{f.reason}</p>
            </div>
            <span className="font-mono text-sm text-amber-400">{f.score.toFixed(2)}</span>
          </li>
        );
      })}
    </ul>
  );
}
