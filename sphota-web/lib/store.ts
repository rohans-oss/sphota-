// The current job, shared by all four app pages (guide F15).
import { create } from "zustand";
import type { CompressResult, DoneEvent, Mode } from "./types";

export type StageState = { status: "waiting" | "running" | "done"; ms?: number };

type State = {
  file?: File;
  previewUrl?: string;
  result?: CompressResult;
  done?: DoneEvent;
  denoise?: { step: number; total: number; preview_url: string };
  stages: Record<string, StageState>;
  presetMode?: Mode;
  set: (p: Partial<State>) => void;
  mark: (stage: string, status: "running" | "done", ms?: number) => void;
  reset: () => void;
};

export const useJob = create<State>((set) => ({
  stages: {},
  set: (p) => set(p),
  mark: (stage, status, ms) => set((s) => ({ stages: { ...s.stages, [stage]: { status, ms } } })),
  reset: () => set({ result: undefined, done: undefined, denoise: undefined, stages: {} }),
}));
