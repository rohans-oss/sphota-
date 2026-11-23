# Sphoṭa web (frontend)

The website for the Sphoṭa / TrueSight codec, built from `Sphota_Frontend_Build_Guide.docx` in **Style 5: retro sci-fi**
(mocha brown, champagne gold, burnt amber). Next.js 16 (App Router) + TypeScript + Tailwind v4 + GSAP + Motion.

This folder is **local only**: it is not in the `sphota` GitHub repo and has no git remote.

## Run it

```bash
cd D:\sphota-web
npm install          # first time only
npm run dev          # http://localhost:3000
```

Production check: `npm run build && npm run start`.

## Real backend (default)

The pages call `/api/*` on their own origin; `next.config.ts` proxies that to the Python service in the main repo
(`SPHOTA_API`, default `http://127.0.0.1:8000`). Start both:

```bash
# 1. the model service (D:\diffusion\sphota): real codecs + diffusion model from experiments/
D:\diffusionenv\Scripts\python.exe -m truesight.app.api          # http://127.0.0.1:8000, GET /api/health shows what loaded
# 2. this site
npm run dev                                                          # http://localhost:3000
```

Every number on the site is measured on the real `.sphota` file the server writes (detect, encode, entropy coding, guided
diffusion decode, JPEG at the same size, OCR / face checks). Not available until their artifacts are trained and exported to
`experiments/serving/`: the mode **recommender** (so no "Recommended" badge) and the **Explain** page (it answers with the reason).
Trust uses the measured seed disagreement and bin-violation signals; the calibrated risk model replaces them when exported.

## Demo mode (no backend needed)

Open any page with `?demo=1` (remembered for the session) or set `NEXT_PUBLIC_DEMO=1` in `.env.local`: every page then runs from
the JSON and images in `public/demo` and `public/proof`. **Those numbers and the medicine-box images are illustration values.**
The API contract is in `lib/types.ts` and `lib/api.ts`.

## Pages

| Route | What it is |
|---|---|
| `/` | Loader, hero, tagline, pinned How it works, proof with split view, FAQ, call to action, footer |
| `/compress` | Drop a photo or pick a sample, choose mode/network/size, watch the live pipeline |
| `/compare` | One seed, four decoders, guidance and step controls, synchronised 3x hover zoom |
| `/trust` | Uncertainty heat map, bin-consistency gauge, flagged regions, safe mode |
| `/explain` | Decision tree path, association rules, kNN neighbours, predicted vs actual |
| `/tokens` | Hidden design-system test sheet |
| any other URL | 404 test card |

## Regenerating the demo images

`python tools/make_demo_assets.py` (needs Pillow and numpy) rebuilds everything in `public/demo`, `public/proof`,
`public/samples`, the hero photo, favicon and og image.

## Notes

- Reduced motion is honoured from the OS setting and the footer toggle (loader, smooth scroll, scrubbing and loops are turned off).
- The guide's `hero.avif` is `hero.webp` here (same role, widely supported).
- Dev checks: `npx tsc --noEmit` and `npm run lint` both pass.
