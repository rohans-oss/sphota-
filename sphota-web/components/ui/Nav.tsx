"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "motion/react";
import { MobileMenu } from "./MobileMenu";

export const APP_LINKS: [string, string][] = [
  ["/compress", "Compress"], ["/compare", "Compare"], ["/trust", "Trust"], ["/explain", "Explain"],
];
export const LANDING_LINKS: [string, string][] = [
  ["#how", "The codec"], ["#proof", "Trust report"], ["#faq", "Research"], ["#cta", "Atlas"],
];

export function Nav({ right, landing }: { right?: React.ReactNode; landing?: boolean }) {
  const path = usePathname();
  const links = landing ? LANDING_LINKS : APP_LINKS;
  return (
    <header className="nav-shell no-print fixed inset-x-0 top-0 z-50 flex items-center px-6 pt-5 md:px-14 md:pt-7">
      <Link href="/" className="font-logo text-[28px] text-gold-100">Sphoṭa</Link>
      <nav aria-label="Main" className="mx-auto hidden gap-1.5 rounded-full border border-gold-200/25 p-1.5 backdrop-blur-md md:flex">
        {links.map(([href, label]) => {
          const active = !landing && path.startsWith(href);
          return (
            <Link
              key={href} href={href} aria-current={active ? "page" : undefined}
              className={"relative rounded-full px-4 py-2 text-[15px] font-medium transition-colors duration-300 " + (active ? "text-mocha-900" : "text-gold-200 hover:text-gold-100")}
            >
              {active && <motion.span layoutId="nav-pill" className="absolute inset-0 rounded-full bg-gold-200" transition={{ type: "spring", bounce: 0.15, duration: 0.5 }} />}
              <span className="relative">{label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="ml-auto hidden md:block">
        {landing ? <Link href="/compress" className="btn-ghost">Open the demo</Link> : right}
      </div>
      <MobileMenu links={landing ? [...LANDING_LINKS, ["/compress", "Open the demo"]] : [["/", "Home"], ...APP_LINKS]} />
    </header>
  );
}
