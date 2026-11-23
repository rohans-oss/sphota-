import { ViewTransition } from "react";
import { Nav } from "@/components/ui/Nav";
import { JobChip } from "@/components/app/JobChip";
import { ScrollTop } from "@/components/app/ScrollTop";

// The four app pages share this layout: app nav + status chip, native scrolling (no Lenis),
// and a CRT switch-off / power-on page transition (guide F14).
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Nav right={<JobChip />} />
      <ScrollTop />
      <ViewTransition>
        <main id="main" className="min-h-[100svh] px-6 pb-14 pt-[96px] md:px-14 md:pt-[104px]">{children}</main>
      </ViewTransition>
    </>
  );
}
