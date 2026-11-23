"use client";

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
      className="flex min-h-11 items-center gap-3 text-sm font-medium"
    >
      <span className={"relative h-7 w-12 rounded-full transition-colors duration-300 " + (checked ? "bg-amber-400" : "bg-mocha-900")}>
        <span className={"absolute top-1 size-5 rounded-full bg-gold-100 transition-all duration-300 " + (checked ? "left-6" : "left-1")} />
      </span>
      {label}
    </button>
  );
}
