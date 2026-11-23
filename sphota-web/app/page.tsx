import { Nav } from "@/components/ui/Nav";
import { SmoothScroll } from "@/components/motion/SmoothScroll";
import { Loader } from "@/components/landing/Loader";
import { Hero } from "@/components/landing/Hero";
import { Tagline } from "@/components/landing/Tagline";
import { HowItWorks } from "@/components/landing/HowItWorks";
import { Proof } from "@/components/landing/Proof";
import { Faq } from "@/components/landing/Faq";
import { Cta } from "@/components/landing/Cta";
import { Footer } from "@/components/landing/Footer";

// Landing page (guide 7.2 to 7.5): loader, hero, tagline, How it works, proof, FAQ, call to action, footer.
export default function Home() {
  return (
    <SmoothScroll>
      <Loader />
      <Nav landing />
      <main id="main">
        <Hero />
        <Tagline />
        <HowItWorks />
        <Proof />
        <Faq />
        <Cta />
      </main>
      <Footer />
    </SmoothScroll>
  );
}
