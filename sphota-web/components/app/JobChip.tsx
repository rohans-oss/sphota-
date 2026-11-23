"use client";
import { useJob } from "@/lib/store";

// Status chip in the app nav: shows the current job.
export function JobChip() {
  const result = useJob((s) => s.result);
  const done = useJob((s) => s.done);
  const stages = useJob((s) => s.stages);
  const running = Object.entries(stages).find(([, v]) => v.status === "running")?.[0];
  const label = done ? `Seed ${(result!.bytes / 1000).toFixed(1)} KB · ${result!.mode}` : running ? `Running · ${running}` : result ? "Seed ready" : "No photo yet";
  return (
    <span className="chip" data-active={!!done} aria-live="polite">
      <span className={"mr-2 inline-block size-2 rounded-full " + (running ? "rec-dot bg-amber-400" : done ? "bg-olive" : "bg-gold-400")} />
      {label}
    </span>
  );
}
