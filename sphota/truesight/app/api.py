"""FastAPI service of the website (guide section 8.3). The contract the pages expect (see sphota-web/lib/api.ts):

    POST /api/compress        multipart: file, mode, network, target_kb  -> CompressResult (the file is encoded for real)
    GET  /api/jobs/{id}/events  Server-Sent Events: detect..pack, denoise previews, done  (replayed from the start to every client)
    POST /api/decode          json: job_id, cfg, steps, seed, safe_mode  -> {job_id} of a new decode run of the same file
    GET  /api/compare/{id}?cfg=&steps=   JPEG / codec only / plain diffusion / Sphoṭa, measured
    GET  /api/explain/{id}    decision path of the mode recommender (503 until its artifacts exist)
    GET  /api/jobs/{id}/files/{name}     images and the .sphota file of a session
    GET  /api/health          models loaded?, missing files, notes

Run:  python -m truesight.app.api   (or: uvicorn truesight.app.api:app --port 8000)
"""
import json
import mimetypes
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .config import MODES, NETWORKS, Settings
from .engine import Engine, Unavailable
from .jobs import JobStore

settings = Settings()
store = JobStore(settings.jobs_dir, settings.job_ttl_s)
engine = Engine(settings)
app = FastAPI(title="Sphota API")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors, allow_methods=["*"], allow_headers=["*"])


def _job(jid):
    job = store.get(jid)
    if job is None:
        raise HTTPException(404, "Unknown or expired session. Compress the photo again.")
    return job


def _root(jid):
    job = _job(jid)
    return job.parent if job.parent is not None else job


@app.on_event("startup")
def _startup():
    def sweep():
        while True:
            time.sleep(600)
            store.sweep()

    threading.Thread(target=sweep, daemon=True).start()
    if settings.preload:
        def load():
            try:
                engine.load()
            except Exception as e:
                print(f"[sphota] preload failed: {e}", flush=True)
        threading.Thread(target=load, daemon=True).start()


@app.get("/api/health")
def health():
    return engine.status()


@app.post("/api/compress")
async def compress(file: UploadFile = File(...), mode: str = Form("small"), network: str = Form("4G"), target_kb: float = Form(0)):
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {', '.join(MODES)}")
    if network not in NETWORKS:
        raise HTTPException(422, f"network must be one of {', '.join(NETWORKS)}")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1_000_000:
        raise HTTPException(413, f"The photo is larger than {settings.max_upload_mb} MB.")
    job = store.create()
    return await _run(lambda: _compress(job, data, mode, network, target_kb))


async def _run(fn):
    import anyio
    try:
        return await anyio.to_thread.run_sync(fn)
    except Unavailable as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


def _compress(job, data, mode, network, target_kb):
    result = engine.compress(job, data, mode, network, target_kb)
    engine.start_decode(job)                       # the first decode runs in the background; its progress is streamed
    return result


@app.get("/api/jobs/{jid}/events")
def events(jid: str):
    job = _job(jid)

    def gen():
        i = 0
        while True:
            batch = job.wait(i, 15)
            if not batch:
                yield ": ping\n\n"
                continue
            for e in batch:
                yield "data: " + json.dumps(e) + "\n\n"
                i += 1
            if job.finished and i >= len(job.events):
                return

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class DecodeBody(BaseModel):
    job_id: str
    cfg: float = 1.5
    steps: int = 25
    seed: int = 7
    safe_mode: bool = False


@app.post("/api/decode")
def decode(body: DecodeBody):
    root = _root(body.job_id)
    if not (1 <= body.steps <= 200) or not (0 <= body.cfg <= 10):
        raise HTTPException(422, "steps must be 1..200 and cfg 0..10")
    run = store.create_run(root)
    engine.start_decode(run, cfg=body.cfg, steps=body.steps, seed=body.seed, safe_mode=body.safe_mode)
    return {"job_id": run.id}


@app.get("/api/compare/{jid}")
async def compare(jid: str, cfg: float = Query(1.5), steps: int = Query(25)):
    root = _root(jid)
    return await _run(lambda: engine.compare(root, cfg, steps))


@app.get("/api/explain/{jid}")
async def explain(jid: str):
    root = _root(jid)
    return await _run(lambda: engine.explain(root))


@app.get("/api/jobs/{jid}/files/{name}")
def files(jid: str, name: str):
    root = _root(jid)
    try:
        path = root.path(name)
    except ValueError:
        raise HTTPException(400, "bad file name")
    if not path.is_file():
        raise HTTPException(404, "No such file")
    kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=kind, filename=name if name.endswith(".sphota") else None,
                        headers={"Cache-Control": "private, max-age=3600"})


if __name__ == "__main__":
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
