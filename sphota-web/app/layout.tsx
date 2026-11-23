import type { Metadata } from "next";
import { Fraunces, Shrikhand, Space_Grotesk, Noto_Sans_Devanagari } from "next/font/google";
import { GeistMono } from "geist/font/mono";
import { Grain } from "@/components/retro/Grain";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"], style: ["normal", "italic"], axes: ["SOFT", "WONK", "opsz"], variable: "--font-fraunces",
});
const shrikhand = Shrikhand({ subsets: ["latin"], weight: "400", variable: "--font-shrikhand" });
const grotesk = Space_Grotesk({ subsets: ["latin"], variable: "--font-grotesk" });
const deva = Noto_Sans_Devanagari({ subsets: ["devanagari"], weight: "500", variable: "--font-noto-deva" });

export const metadata: Metadata = {
  title: "Sphoṭa · The whole picture, in a seed",
  description:
    "Sphoṭa sends a tiny seed instead of a heavy photo. A diffusion model grows it back, and the doses, names and faces arrive exactly as they left.",
  icons: { icon: "/favicon.svg" },
  openGraph: {
    title: "Sphoṭa · The whole picture, in a seed",
    description: "A faithful generative image codec: tiny files, and nothing that matters ever changes.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const fonts = [fraunces, shrikhand, grotesk, deva].map((f) => f.variable).join(" ");
  return (
    <html lang="en" className={fonts + " " + GeistMono.variable} suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: "try{if(localStorage.getItem('motion')==='reduced')document.documentElement.dataset.motion='reduced';if(sessionStorage.getItem('seen'))document.documentElement.dataset.seen='1'}catch(e){}",
          }}
        />
      </head>
      <body>
        <a href="#main" className="skip-link">Skip to content</a>
        {children}
        <Grain />
      </body>
    </html>
  );
}
