"use client";
import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Minus, Plus } from "@phosphor-icons/react";

const FAQ = [
  { q: "What is a seed?", a: "The entropy-coded VAE latent plus the exact pixels of protected regions." },
  { q: "Can diffusion change the text on my photo?", a: "Not in protected regions: they are sent exactly, and bin-consistency guidance keeps the rest close to the seed." },
  { q: "How is this different from JPEG or WebP at the same size?", a: "At 1 to 3 KB they turn into blocks. Sphoṭa regrows detail. See the proof section above." },
  { q: "What happens when the model is unsure?", a: "The witness flags it on the Trust page, and safe mode shows honest blur instead." },
  { q: "Does it need a GPU on my phone?", a: "No. Decoding runs on the server for the demo; on-device decoding is future work." },
];

// One item open at a time; height animated by Motion (guide F13).
export function Faq() {
  const [open, setOpen] = useState(0);
  return (
    <section id="faq" className="px-6 py-20 md:px-14">
      <p className="eyebrow">Questions</p>
      <h2 className="h2-serif mt-3">Asked <em>often.</em></h2>
      <div className="mt-10 max-w-[820px]">
        {FAQ.map((q, i) => (
          <div key={q.q} className="border-t border-gold-200/20 last:border-b">
            <button className="flex min-h-11 w-full items-center justify-between gap-6 py-5 text-left text-[19px] font-medium" aria-expanded={open === i} onClick={() => setOpen(open === i ? -1 : i)}>
              {q.q}
              <motion.span animate={{ rotate: open === i ? 180 : 0 }}>{open === i ? <Minus /> : <Plus />}</motion.span>
            </button>
            <AnimatePresence initial={false}>
              {open === i && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }} className="overflow-hidden">
                  <p className="max-w-[640px] pb-5 text-gold-400">{q.a}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </section>
  );
}
