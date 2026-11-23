"use client";
import { useRef } from "react";
import { gsap, useGSAP } from "@/lib/motion";
import { prefersReduced } from "@/lib/device";
import type { TreeEdge, TreeNode } from "@/lib/types";

// Layered layout, no graph library; the taken path draws itself node by node (guide F19).
const X = (c: number) => 60 + c * 740;
const Y = (d: number) => 60 + d * 140;
const widthOf = (n: TreeNode) => Math.max(120, n.label.length * 8.4 + 36);

export function DecisionTree({ nodes, edges }: { nodes: TreeNode[]; edges: TreeEdge[] }) {
  const svg = useRef<SVGSVGElement>(null);
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

  useGSAP(
    () => {
      if (prefersReduced()) return;
      const paths = gsap.utils.toArray<SVGPathElement>(".edge.taken");
      const taken = gsap.utils.toArray<SVGGElement>(".node.taken");
      const tl = gsap.timeline({ delay: 0.3 });
      taken.forEach((n, i) => {
        tl.from(n, { opacity: 0.25, scale: 0.94, transformOrigin: "50% 50%", duration: 0.35 }, i * 0.45);
        if (paths[i]) tl.fromTo(paths[i], { drawSVG: "0%" }, { drawSVG: "100%", duration: 0.45, ease: "power1.inOut" }, i * 0.45 + 0.1);
      });
      tl.from(".edge-label", { opacity: 0, duration: 0.3, stagger: 0.15 }, 0.6);
      tl.to(".node.chosen rect", { fill: "#E07A2E", duration: 0.4 });
    },
    { scope: svg, dependencies: [nodes] },
  );

  return (
    <svg ref={svg} viewBox="0 0 860 540" className="w-full" role="img" aria-label="Decision tree that chose the mode">
      <defs>
        <pattern id="plotter" width="40" height="40" patternUnits="userSpaceOnUse">
          <path d="M40 0H0V40" fill="none" stroke="#E9D4A4" strokeOpacity=".07" />
        </pattern>
      </defs>
      <rect width="860" height="540" fill="url(#plotter)" />
      {edges.map((e, i) => {
        const a = byId[e.from], b = byId[e.to];
        const x1 = X(a.col), y1 = Y(a.depth), x2 = X(b.col), y2 = Y(b.depth), my = (y1 + y2) / 2;
        return (
          <g key={i}>
            <path className={"edge " + (e.taken ? "taken" : "")} d={`M${x1} ${y1 + 22} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2 - 22}`}
              fill="none" stroke={e.taken ? "#E07A2E" : "#86593F"} strokeWidth={e.taken ? 3 : 1.5} strokeDasharray={e.taken ? undefined : "5 6"} />
            {e.label && (
              <text className={"edge-label " + (e.taken ? "" : "opacity-60")} x={(x1 + x2) / 2 + (x2 >= x1 ? 8 : -8)} y={my - 4} textAnchor={x2 >= x1 ? "start" : "end"}
                fontFamily="var(--font-geist-mono)" fontSize="12" fill={e.taken ? "#E07A2E" : "#BFA57A"}>{e.label}</text>
            )}
          </g>
        );
      })}
      {nodes.map((n) => {
        const w = widthOf(n);
        return (
          <g key={n.id} className={"node " + (n.taken ? "taken " : "") + (n.chosen ? "chosen" : "")} transform={`translate(${X(n.col)} ${Y(n.depth)})`}>
            <rect x={-w / 2} y={-22} width={w} height={44} rx={22} fill={n.taken ? "#4A2F22" : "#5A3A2B"} stroke={n.taken ? "#E07A2E" : "#86593F"} strokeWidth={n.taken ? 2 : 1.2} />
            <text textAnchor="middle" y={5} fill="#F5E7C6" fontSize={n.leaf ? 16 : 13}
              fontFamily={n.leaf ? "var(--font-grotesk)" : "var(--font-geist-mono)"} fontWeight={n.leaf ? 600 : 400}>{n.label}</text>
          </g>
        );
      })}
    </svg>
  );
}
