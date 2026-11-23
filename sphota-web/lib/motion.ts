"use client";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { SplitText } from "gsap/SplitText";
import { CustomEase } from "gsap/CustomEase";
import { Flip } from "gsap/Flip";
import { DrawSVGPlugin } from "gsap/DrawSVGPlugin";
import { ScrambleTextPlugin } from "gsap/ScrambleTextPlugin";
import { useGSAP } from "@gsap/react";

gsap.registerPlugin(ScrollTrigger, SplitText, CustomEase, Flip, DrawSVGPlugin, ScrambleTextPlugin, useGSAP);
CustomEase.create("sphota", "0.22,1,0.36,1"); // arrivals
CustomEase.create("wipe", "0.76,0,0.24,1"); // big masks and page wipes
gsap.defaults({ ease: "sphota", duration: 0.9 });

// CRT power-on and switch-off (guide F5)
export function crtOn(el: gsap.TweenTarget, delay = 0) {
  return gsap.fromTo(
    el,
    { scaleY: 0.01, scaleX: 0.6, filter: "brightness(3)" },
    { scaleY: 1, scaleX: 1, filter: "brightness(1)", duration: 0.7, ease: "expo.out", delay },
  );
}
export function crtOff(el: gsap.TweenTarget) {
  return gsap
    .timeline()
    .to(el, { scaleY: 0.01, filter: "brightness(3)", duration: 0.25, ease: "power2.in" })
    .to(el, { scaleX: 0, duration: 0.15, ease: "power2.in" });
}

export { gsap, ScrollTrigger, SplitText, Flip, useGSAP };
