import Link from "next/link";
import { TestCard } from "@/components/retro/TestCard";

// 404: no signal (guide F21).
export default function NotFound() {
  return (
    <main id="main" className="grid min-h-[100svh] items-center gap-12 px-6 py-16 md:grid-cols-[minmax(0,560px)_1fr] md:gap-20 md:px-[8vw]">
      <TestCard />
      <div>
        <p className="eyebrow">Error 404 · no signal</p>
        <h1 className="display mt-4 !text-[clamp(44px,5vw,72px)]">This page never <em>left the sender.</em></h1>
        <p className="mt-5 text-lg text-gold-200">The link may be old, or the seed was never sent. Head back and try a real transmission.</p>
        <div className="mt-8 flex flex-wrap gap-4">
          <Link href="/" className="btn-primary">Back to home</Link>
          <Link href="/compress" className="btn-ghost">Compress a photo</Link>
        </div>
      </div>
    </main>
  );
}
