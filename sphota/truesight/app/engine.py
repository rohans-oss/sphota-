"""The engine of the web service: it owns the trained codecs, the diffusion model and the detectors, and runs the REAL Sphoṭa pipeline
for the website (detect -> encode -> real .sphota bitstream -> decode with guidance -> honesty signals). Nothing here is a mock:
every number the site shows is measured on the files this module writes.

Heavy work is serialised with one lock (one model, one device). Everything is overridable through truesight.app.config.
"""
import io
import math
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import DEFAULT_SEED, LAMBDAS, MODE_LAMBDA, MODES, ROI_LAMBDA

PREVIEWS = 5                      # denoise frames streamed to the page per decode
PREVIEW_SIDE = 384                # longest side of a preview frame
FLAG_THRESHOLD = 0.5              # uncertainty score from which a patch is flagged
MAX_FLAGS = 4


class Unavailable(RuntimeError):
    """A feature whose trained artifact is not present (reported to the page as a 503 with the reason)."""


def _log(msg):
    print(f"[sphota] {msg}", flush=True)


def _to_t(pil, device="cpu"):
    import torch
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None].to(device)


def _to_pil(t):
    return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype("uint8"))


class Engine:
    def __init__(self, settings):
        self.s = settings
        self.lock = threading.RLock()            # one heavy job at a time
        self._load_lock = threading.Lock()
        self.ready, self.error = False, None
        self.codecs, self.unet, self.sched = {}, None, None
        self.text_det = self.face_det = self.ocr = self.embedder = None
        self.artifacts = None
        self.notes = []                          # things that did not load (detectors, metrics), shown by /api/health

    # ------------------------------------------------------------------------------------------------ loading
    def status(self):
        s = self.s
        return {"ready": self.ready, "error": self.error, "device": s.device, "max_pixels": s.max_pixels,
                "missing": [str(p.relative_to(s.experiments.parent)) if s.experiments.parent in p.parents else str(p)
                            for p in s.required_files() if not p.exists()],
                "diffusion_checkpoint": s.diffusion_path.name, "notes": list(self.notes),
                "artifacts_missing": list(self.artifacts.missing) if self.artifacts is not None else None}

    def load(self):
        """Load every checkpoint once. Raises Unavailable with the list of missing files."""
        with self._load_lock:
            if self.ready:
                return
            import torch
            from ..codec.train_codec import load_codec
            from ..codec.vae_codec import LAMBDAS as CODEC_LAMBDAS
            from ..diffusion.schedule import Schedule
            from ..diffusion.unet import make_unet
            assert tuple(CODEC_LAMBDAS) == tuple(LAMBDAS), "config.LAMBDAS differs from truesight.codec.vae_codec.LAMBDAS"
            s = self.s
            miss = [p for p in s.required_files() if not p.exists()]
            if miss:
                self.error = "missing checkpoints: " + ", ".join(p.name + " (" + p.parent.name + ")" for p in miss)
                raise Unavailable(self.error)
            dev = torch.device(s.device)
            t0 = time.time()
            for i in sorted(set(MODE_LAMBDA.values()) | {ROI_LAMBDA}):
                net, cfg = load_codec(str(s.codec_path(i)), dev)
                net.prepare_for_coding()
                self.codecs[i] = net
            ck = torch.load(str(s.diffusion_path), map_location="cpu", mmap=True, weights_only=False)   # mmap: the file also holds optimiser state
            ucfg = ck["config"]
            unet = make_unet(**ucfg["unet"])
            unet.load_state_dict(ck["ema"])
            del ck
            self.unet, self.sched = unet.to(dev).eval(), Schedule(ucfg["T"], ucfg["kind"], device=dev)
            self.gk = {k: v for k, v in s.guidance().items() if k not in ("w", "steps")}
            from .artifacts import load_artifacts
            try:
                self.artifacts = load_artifacts(s)
            except Exception as e:                                       # the optional layers must never stop the codec
                from .artifacts import Artifacts
                self.artifacts = Artifacts(missing=["recommender", "size_model", "risk_model", "failure_rules"])
                self.notes.append(f"intelligence artifacts not loaded: {type(e).__name__}: {e}")
            self.ready, self.error = True, None
            _log(f"models loaded on {dev} in {time.time() - t0:.1f}s (diffusion: {s.diffusion_path.name})")

    # ------------------------------------------------------------------------------------------------ detectors / metrics (lazy)
    def _detect(self, pil):
        """(text boxes, face boxes) as (x0, y0, x1, y1). A detector that cannot load is reported in ``notes`` and skipped."""
        from ..protect.detect import FaceDetector, TextDetector
        text, faces = [], []
        try:
            self.text_det = self.text_det or TextDetector(gpu=self.s.device == "cuda")
            text = [tuple(b) for b in self.text_det.detect(pil)]
        except Exception as e:
            self._note("text detector", e)
        try:
            self.face_det = self.face_det or FaceDetector(device=self.s.device)
            faces = [tuple(b) for b in self.face_det.detect(pil)]
        except Exception as e:
            self._note("face detector", e)
        return text, faces

    def _note(self, what, e):
        msg = f"{what} unavailable: {type(e).__name__}: {e}"
        if msg not in self.notes:
            self.notes.append(msg)
            _log(msg)

    def _lpips(self, a, b):
        try:
            from ..metrics import lpips_dist
            return float(lpips_dist(a.cpu(), b.cpu()))
        except Exception as e:
            self._note("LPIPS", e)
            return None

    # ------------------------------------------------------------------------------------------------ compress
    def compress(self, job, data, mode, network, target_kb):
        """Run detect -> recommend -> encode -> entropy -> pack on an uploaded photo. Emits the five stage events and returns the
        CompressResult dict the page expects. Raises ValueError for a file that is not a usable image."""
        import torch
        from ..codec import bitstream as bs
        from ..protect import roi
        self.load()
        s = self.s
        try:
            pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except Exception:
            raise ValueError("That file is not an image the server can read (try JPEG, PNG or WebP).")
        W0, H0 = pil.size
        if min(W0, H0) < 32:
            raise ValueError("The photo is too small (at least 32 pixels on each side).")
        if s.max_pixels and W0 * H0 > s.max_pixels:
            f = math.sqrt(s.max_pixels / float(W0 * H0))
            pil = pil.resize((max(32, int(W0 * f)), max(32, int(H0 * f))), Image.LANCZOS)
        W, H = pil.size
        pil.save(job.path("original.png"))
        warnings, timings = [], {}
        mode_i = MODE_LAMBDA[mode]
        lo, hi = self.codecs[mode_i], self.codecs[ROI_LAMBDA]

        def stage(name, fn):
            job.emit({"stage": name, "status": "running"})
            t = time.time()
            out = fn()
            ms = (time.time() - t) * 1000
            timings[name] = round(ms / 1000, 3)
            job.emit({"stage": name, "status": "done", "ms": round(ms)})
            return out

        with self.lock:
            text, faces = stage("detect", lambda: self._detect(pil))
            boxes = roi.plan_boxes(text + faces, H, W)
            rec = stage("recommend", lambda: self._recommend(pil, network))
            x = _to_t(pil, next(lo.parameters()).device)
            g = self.s.guidance()
            guidance = int(round(g["w"] * 10))

            def encode():
                with torch.no_grad():
                    lo.g_a(bs.pad_to_multiple(x, lo.DOWN))      # the analysis transform (timed apart from the entropy coder)
            t = time.time()
            stage("encode", encode)
            t_encode = time.time() - t
            blob = [None]

            def entropy():
                blob[0], _ = roi.encode_protected(lo, hi, x, boxes, mode=mode_i, seed=DEFAULT_SEED, guidance=guidance)
            stage("entropy", entropy)
            blob = blob[0]

            def pack():
                job.path("seed.sphota").write_bytes(blob)
            stage("pack", pack)
        job.meta.update(mode=mode, mode_i=mode_i, H=H, W=W, boxes=boxes, text=text, faces=faces, network=network,
                        bytes=len(blob), original_bytes=len(data), cfg=g["w"], steps=s.steps_for(mode, g), seed=DEFAULT_SEED)
        if target_kb and len(blob) > target_kb * 1000:
            warnings.append(f"The seed is {len(blob) / 1000:.1f} KB, above the {target_kb:.0f} KB target: choose a smaller mode.")
        if not (text or faces) and self.notes:
            warnings.append("No text or faces were found; " + "; ".join(self.notes))
        for stage_name in ("detect", "recommend", "encode", "entropy", "pack"):
            timings.setdefault(stage_name, 0.0)
        return {"job_id": job.id, "bytes": len(blob), "original_bytes": len(data), "bpp": round(8 * len(blob) / (H * W), 4),
                "mode": mode, "recommended_mode": rec, "boxes": [self._norm_box(b, W, H, faces) for b in boxes],
                "timings": timings, "seed_url": job.url("seed.sphota"), "warnings": warnings,
                "size": {"width": W, "height": H}}

    def _recommend(self, pil, network):
        """The mode the trained recommender picks, or None while its artifacts (Step 14) are not exported."""
        a = self.artifacts
        if a is None or a.recommender is None:
            return None
        try:
            from ..analysis.features import image_features
            feats = image_features(pil)
            ctx = {**feats}
            mode, _ = a.recommender.recommend(ctx, network)[:2]
            return mode if mode in MODES else None
        except Exception as e:
            self._note("mode recommender", e)
            return None

    @staticmethod
    def _norm_box(b, W, H, faces):
        x, y, w, h = b
        cx, cy = x + w / 2, y + h / 2
        is_face = any(fx0 <= cx <= fx1 and fy0 <= cy <= fy1 for fx0, fy0, fx1, fy1 in faces)
        return {"x": round(x / W, 4), "y": round(y / H, 4), "w": round(w / W, 4), "h": round(h / H, 4),
                "kind": "face" if is_face else "text", "label": "Face" if is_face else "Text"}

    # ------------------------------------------------------------------------------------------------ decoding
    def _sphota(self, job, cfg, steps, seed, guided=True, use_roi=True, on_step=None):
        """Decode the job's real .sphota file. Returns (SphotaOutput, prot, lo, seconds). ``on_step(i, total, x0_01)`` is called after
        every denoising step (x0 estimate in [0, 1])."""
        from ..diffusion import guidance as G
        from ..protect import roi
        blob = job.path("seed.sphota").read_bytes()
        lo, hi = self.codecs[job.meta["mode_i"]], self.codecs[ROI_LAMBDA]
        t0 = time.time()
        prot = roi.decode_protected(lo, hi, blob)
        if steps == 0:                                   # "exact": the codec alone, no generative step
            out = G.SphotaOutput(x=prot.coarse.clamp(0, 1), x_diffusion=prot.coarse, coarse=prot.coarse, prot=prot)
            if use_roi and prot.mask.sum() > 0:
                out.x = roi.paste(prot.coarse, prot.canvas.to(prot.coarse.device), prot.soft_mask.to(prot.coarse.device)).clamp(0, 1)
            return out, prot, lo, time.time() - t0
        H, W = prot.coarse.shape[-2:]
        original = G.diffusion_decode
        counter = {"i": 0}

        class Tap:
            def __init__(self, inner):
                self.inner = inner

            def __call__(self, x0, t):
                out = self.inner(x0, t) if self.inner is not None else x0
                counter["i"] += 1
                if on_step is not None:
                    on_step(counter["i"], steps, ((out + 1) / 2).clamp(0, 1)[..., :H, :W])
                return out

        def patched(unet, sched, coarse, **kw):
            kw["guide"] = Tap(kw.get("guide"))
            return original(unet, sched, coarse, **kw)

        with self.lock:
            G.diffusion_decode = patched
            try:
                out = G.sphota_decode(lo, hi, self.unet, self.sched, prot=prot, w=cfg, steps=steps, seed=seed, guided=guided,
                                      use_roi=use_roi, **self.gk)
            finally:
                G.diffusion_decode = original
        return out, prot, lo, time.time() - t0

    def run_decode(self, job, cfg=None, steps=None, seed=None, safe_mode=False):
        """Decode the session's file (called in a thread). Streams ``denoise`` previews, computes the honesty signals, writes the
        images and ends the job with a ``done`` event (or ``error``)."""
        try:
            self.load()
            m = job.parent.meta if job.parent is not None else job.meta
            job.meta = m                                           # a re-decode shares the facts of its session
            cfg = m["cfg"] if cfg is None else float(cfg)
            steps = m["steps"] if steps is None else int(steps)
            seed = m["seed"] if seed is None else int(seed)
            if self.s.cpu_steps > 0:
                steps = min(steps, self.s.cpu_steps)
            every = max(1, math.ceil(steps / PREVIEWS))

            def on_step(i, total, x01):
                if i % every == 0 and i < total:
                    name = f"{job.id}_denoise_{i}.webp"
                    im = _to_pil(x01.cpu())
                    im.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
                    im.save(job.path(name), quality=80)
                    job.emit({"stage": "denoise", "step": i, "total": total, "preview_url": job.url(name)})

            with self.lock:
                out, prot, lo, secs = self._sphota(job, cfg, steps, seed, on_step=on_step)
                decodes = [out.x]
                n = self.s.n_seeds if steps > 0 else 1
                for k in range(1, n):                           # other seeds: only for the uncertainty map
                    decodes.append(self._sphota(job, cfg, steps, seed + k)[0].x)
                unc = self._uncertainty(job, lo, prot, decodes)
                from ..diffusion.guidance import consistency_rate
                cons = consistency_rate(lo, out.x, prot.y_hat)
            final = out.x
            if safe_mode and unc["flags_grid"] is not None:
                from ..analysis.hallucination import safe_mode as _safe
                final = _safe(out.x.cpu(), prot.coarse.cpu(), unc["flags_grid"], prot.mask.cpu() if len(prot.boxes) else None)
            _to_pil(final).save(job.path(f"{job.id}_image.png"))
            _to_pil(prot.coarse).save(job.path(f"{job.id}_coarse.png"))
            unc["image"].save(job.path(f"{job.id}_uncertainty.png"))
            m.setdefault("decodes", {})[(round(cfg, 3), steps, seed)] = {"sphota": out.x.cpu(), "seconds": secs, "consistency": cons}
            job.emit({"stage": "done", "image_url": job.url(f"{job.id}_image.png"), "coarse_url": job.url(f"{job.id}_coarse.png"),
                      "uncertainty_url": job.url(f"{job.id}_uncertainty.png"), "consistency": round(float(cons), 4),
                      "flagged": unc["flags"], "decode_s": round(secs, 2), "steps": steps, "cfg": cfg, "seed": seed})
        except Exception as e:                              # the page shows the message; the traceback goes to the server log
            import traceback
            traceback.print_exc()
            job.fail(f"{type(e).__name__}: {e}")

    def start_decode(self, job, **kw):
        t = threading.Thread(target=self.run_decode, args=(job,), kwargs=kw, daemon=True, name=f"decode-{job.id}")
        t.start()
        return t

    # ------------------------------------------------------------------------------------------------ honesty signals
    def _uncertainty(self, job, lo, prot, decodes):
        """Per-patch uncertainty from REAL signals of this decode: the disagreement of the seeds (the bits did not decide the content)
        and how far the output leaves the transmitted bins. Protected patches (sent at high quality) are safe by construction.
        This is a measured signal, not the calibrated risk model of Step 13 (its artifact is optional, see artifacts.py)."""
        import torch
        from ..analysis import hallucination as hz
        H, W = decodes[0].shape[-2:]
        ph, pw = H // hz.PATCH, W // hz.PATCH
        if ph == 0 or pw == 0:
            return {"image": Image.new("RGBA", (W, H), (0, 0, 0, 0)), "flags": [], "flags_grid": None}
        xs = [d.detach().cpu() for d in decodes]
        sv = hz.seed_variance_map(xs).reshape(ph, pw) if len(xs) > 1 else torch.zeros(ph, pw)
        bv = hz.bin_violation_map(lo, xs[0], prot.y_hat).reshape(ph, pw)
        score = 1 - torch.exp(-(sv / 0.03 + bv / 0.05))
        a = self.artifacts
        if a is not None and a.risk is not None:
            try:                                                  # the trained, calibrated model replaces the heuristic when exported
                tab = hz.patch_table(xs, prot.coarse.cpu(), prot.y_hat, lo, job.meta.get("text", []) + job.meta.get("faces", []),
                                     job.meta["boxes"])
                score = torch.as_tensor(a.risk.predict_risk(tab["X"])).reshape(ph, pw).float()
            except Exception as e:
                self._note("risk model", e)
        prot_cov = hz.to_patches(hz.box_mask(H, W, prot.boxes, xywh=True), hz.PATCH).reshape(ph, pw)
        score = score * (prot_cov < 0.5).float()
        flags_grid = (score >= FLAG_THRESHOLD)
        u = hz.upsample_scores(score.reshape(-1).numpy(), (ph, pw), H, W)
        r = (1.5 - (4 * u - 3).abs()).clamp(0, 1)
        g = (1.5 - (4 * u - 2).abs()).clamp(0, 1)
        b = (1.5 - (4 * u - 1).abs()).clamp(0, 1)
        rgba = torch.cat([r, g, b, (0.8 * u).clamp(0, 1)], dim=1)[0].permute(1, 2, 0).numpy()
        heat = Image.fromarray((rgba * 255).round().astype("uint8"), "RGBA")
        flags = self._flags(job, score.numpy(), sv.numpy(), bv.numpy(), flags_grid.numpy(), hz.PATCH, W, H)
        return {"image": heat, "flags": flags, "flags_grid": flags_grid}

    def _flags(self, job, score, sv, bv, grid, patch, W, H):
        from scipy import ndimage
        lab, n = ndimage.label(grid)
        out = []
        for k in range(1, n + 1):
            ys, xs = np.nonzero(lab == k)
            y0, y1, x0, x1 = ys.min() * patch, (ys.max() + 1) * patch, xs.min() * patch, (xs.max() + 1) * patch
            m = lab == k
            sc, s_var, b_viol = float(score[m].max()), float(sv[m].mean()), float(bv[m].mean())
            over_text = any(bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0 for bx0, by0, bx1, by1 in job.meta.get("text", []))
            over_face = any(bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0 for bx0, by0, bx1, by1 in job.meta.get("faces", []))
            if over_text:
                name, icon = "Text the file does not protect", "text"
            elif over_face:
                name, icon = "Face the file does not protect", "texture"
            else:
                name, icon = "Detail the bits did not decide", "texture"
            why = f"Decodes with different seeds disagree here (spread {s_var:.3f})" if s_var / 0.03 >= b_viol / 0.05 else \
                  "The decoded pixels leave the rounding bins of the transmitted latent"
            out.append({"id": f"f{k}", "name": name, "reason": why, "score": round(sc, 3), "icon": icon,
                        "box": {"x": round(x0 / W, 4), "y": round(y0 / H, 4), "w": round((x1 - x0) / W, 4),
                                "h": round((y1 - y0) / H, 4), "kind": "text" if over_text else "face", "label": name}})
        out.sort(key=lambda f: -f["score"])
        return out[:MAX_FLAGS]

    # ------------------------------------------------------------------------------------------------ compare
    def compare(self, job, cfg, steps, seed=None):
        """Four decoders of the same file, measured: JPEG at the same size, the codec alone, plain diffusion and Sphoṭa."""
        self.load()
        m = job.meta
        seed = m["seed"] if seed is None else int(seed)
        if self.s.cpu_steps > 0:
            steps = min(steps, self.s.cpu_steps)
        key = (round(float(cfg), 3), int(steps), seed)
        cache = m.setdefault("compare", {})
        if key in cache:
            return cache[key]
        with self.lock:
            orig = Image.open(job.path("original.png")).convert("RGB")
            dev = next(self.codecs[m["mode_i"]].parameters()).device
            xo = _to_t(orig, dev)
            sph, prot, lo, t_s = self._sphota(job, cfg, steps, seed)
            pln, _, _, t_p = self._sphota(job, cfg, steps, seed, guided=False, use_roi=False)
            jpeg_img, jpeg_q = self._jpeg_at_size(orig, m["bytes"])
            variants = {"jpeg": _to_t(jpeg_img, dev), "coarse": prot.coarse.clamp(0, 1), "plain": pln.x, "sphota": sph.x}
            from ..diffusion.guidance import consistency_rate
            pils = {k: _to_pil(v.cpu()) for k, v in variants.items()}
            tag = f"cmp_{int(round(cfg * 10))}_{steps}_{seed}"
            res, tfr, fir = {}, {}, {}
            for name, img in pils.items():
                img.save(job.path(f"{tag}_{name}.png"))
                lp = self._lpips(xo, variants[name])
                t_res, f_res = self._faith(orig, img)
                tfr[name], fir[name] = t_res, f_res
                res[name] = {"url": job.url(f"{tag}_{name}.png"), "lpips": None if lp is None else round(lp, 4),
                             "verdict": self._verdict(t_res, f_res, name == "jpeg", jpeg_q), "kept": self._kept(t_res, f_res)}
            cons = {k: float(consistency_rate(lo, variants[k], prot.y_hat)) for k in ("plain", "sphota")}
            metrics = {"tfr": {k: self._tfr_value(tfr[k]) for k in ("plain", "sphota")},
                       "fir": {k: self._fir_value(fir[k]) for k in ("plain", "sphota")},
                       "consistency": cons, "decode_s": {"plain": round(t_p, 2), "sphota": round(t_s, 2)}}
            out = {**res, "metrics": metrics}
        cache[key] = out
        return out

    @staticmethod
    def _jpeg_at_size(pil, target_bytes):
        """The JPEG quality whose file is the closest to (and not above) the seed's size; quality 1 if even that is larger."""
        def enc(q):
            b = io.BytesIO()
            pil.save(b, "JPEG", quality=q)
            return b.getvalue()
        lo_q, hi_q, best = 1, 95, 1
        while lo_q <= hi_q:
            q = (lo_q + hi_q) // 2
            if len(enc(q)) <= target_bytes:
                best, lo_q = q, q + 1
            else:
                hi_q = q - 1
        return Image.open(io.BytesIO(enc(best))).convert("RGB"), best

    def _faith(self, orig, dec):
        """(text result, face result) of the faithfulness metrics for one decode; None for a part that cannot be measured."""
        t_res = f_res = None
        try:
            from ..protect.faithfulness import EasyOCRAdapter, text_faithfulness
            self.ocr = self.ocr or EasyOCRAdapter(gpu=self.s.device == "cuda")
            t_res = text_faithfulness(self.ocr, orig, dec)
        except Exception as e:
            self._note("text faithfulness (OCR)", e)
        try:
            from ..protect.detect import FaceDetector
            from ..protect.faithfulness import FaceEmbedder, face_identity
            self.face_det = self.face_det or FaceDetector(device=self.s.device)
            self.embedder = self.embedder or FaceEmbedder(device=self.s.device)
            f_res = face_identity(self.face_det, self.embedder, orig, dec) or None
        except Exception as e:
            self._note("face identity", e)
        return t_res, f_res

    @staticmethod
    def _kept(t_res, f_res):
        return bool((t_res is None or t_res["passed"]) and (f_res is None or all(f["passed"] for f in f_res)))

    @staticmethod
    def _verdict(t_res, f_res, is_jpeg, q):
        parts = []
        if t_res is not None:
            bad = [r for r in t_res["regions"] if r["cer"] > 0.05]
            if bad:
                r = bad[0]
                parts.append(f"Text changed: '{r['original']}' read as '{r['decoded']}'")
            else:
                parts.append(f"Text kept ({t_res['n_regions']} regions)")
        if f_res is not None:
            bad = [f for f in f_res if not f["passed"]]
            parts.append(f"Face identity changed ({len(bad)} of {len(f_res)})" if bad else f"Face identity kept ({len(f_res)})")
        if not parts:
            parts.append("No text or faces to check")
        if is_jpeg:
            parts.append(f"JPEG quality {q}")
        return " · ".join(parts)

    @staticmethod
    def _tfr_value(t_res):
        return None if t_res is None else round(t_res["n_ok"] / t_res["n_regions"], 3)

    @staticmethod
    def _fir_value(f_res):
        return None if not f_res else round(float(np.mean([f["cosine"] for f in f_res])), 3)

    # ------------------------------------------------------------------------------------------------ explain
    def explain(self, job):
        """The decision path / rules / neighbours of the mode recommender (Step 14). Needs experiments/serving/*.joblib."""
        self.load()
        a = self.artifacts
        if a is None or a.recommender is None or a.tree is None or a.knn is None:
            raise Unavailable("The intelligence models (mode recommender, size predictor) are not trained or exported yet: "
                              "experiments/serving/recommender.joblib is missing. Run the Step 14 training and "
                              "scripts/export_serving_models.py, then restart the server.")
        from ..analysis.features import image_features
        from . import explain as ex
        m = job.meta
        orig = Image.open(job.path("original.png")).convert("RGB")
        feats = image_features(orig)
        net = m["network"]
        tree, leaf = ex.tree_view(a.tree, feats, net)
        neigh = []
        for img_id, mode in ex.neighbours_view(a.knn, feats, net):
            neigh.append({"thumb_url": "", "best_mode": mode, "name": str(img_id)})
        size = ex.size_view(a.size, feats, m["H"], m["W"], m["mode_i"]) if a.size else {"kb": m["bytes"] / 1000, "kb_std": 0.0, "lpips": None}
        return {"tree": tree, "rules": ex.rules_view(a.rules, [], MODES[leaf], m["mode"], []), "neighbours": neigh,
                "predicted": size, "actual": {"kb": round(m["bytes"] / 1000, 3), "lpips": None}}
