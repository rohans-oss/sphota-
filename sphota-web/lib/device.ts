export const prefersReduced = () =>
  typeof window !== "undefined" &&
  (window.matchMedia("(prefers-reduced-motion: reduce)").matches ||
    document.documentElement.dataset.motion === "reduced"); // footer toggle (F13)

export const isWeakDevice = () => {
  if (typeof navigator === "undefined") return false;
  const nav = navigator as Navigator & { deviceMemory?: number };
  return (nav.hardwareConcurrency ?? 4) < 4 || (nav.deviceMemory ?? 4) < 4;
};
