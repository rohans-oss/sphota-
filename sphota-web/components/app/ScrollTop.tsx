"use client";
import { usePathname } from "next/navigation";
import { useEffect } from "react";

// Always start a new app page at the top (the shared layout would otherwise keep the old scroll position).
export function ScrollTop() {
  const path = usePathname();
  useEffect(() => { window.scrollTo(0, 0); }, [path]);
  return null;
}
