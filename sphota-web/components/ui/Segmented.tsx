"use client";
import { motion } from "motion/react";
import { useId } from "react";

type Opt<T extends string> = readonly [T, string];

// Segmented control; the active background slides with Motion layoutId.
export function Segmented<T extends string>({
  value, onChange, options, label,
}: { value: T; onChange: (v: T) => void; options: readonly Opt<T>[]; label?: string }) {
  const id = useId();
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap gap-1 rounded-full border border-gold-200/25 p-1">
      {options.map(([v, text]) => {
        const active = v === value;
        return (
          <button
            key={v} type="button" role="radio" aria-checked={active} onClick={() => onChange(v)}
            className={"relative min-h-11 rounded-full px-4 py-2 text-sm font-medium transition-colors duration-300 " + (active ? "text-mocha-900" : "text-gold-200 hover:text-gold-100")}
          >
            {active && <motion.span layoutId={"seg-" + id} className="absolute inset-0 rounded-full bg-gold-200" transition={{ type: "spring", bounce: 0.15, duration: 0.45 }} />}
            <span className="relative">{text}</span>
          </button>
        );
      })}
    </div>
  );
}
