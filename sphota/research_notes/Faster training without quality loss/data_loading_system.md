# Input-pipeline and system-level speedups for image-model training on Kaggle (4 CPU, 2x T4, ~29 GB RAM)

Evidence caveat: web search in this session returned mostly secondary sources. Official Kaggle docs pages (kaggle.com/docs/notebooks, /docs/datasets) are JavaScript-rendered and returned no body text to the fetch tool, so Kaggle limits below are secondary or unverified. No source I found measures PNG decode throughput on 4 cores, GPU PNG decode, or memmap random crops on Kaggle's disk. Where I give numbers that are not cited they are labelled as arithmetic or inference.

## 1. Realistic throughput per option, GPU consumption rate, and best fit for lossless PNG sources

### Takeaway
For exact-pixel PNG sources the decode must stay lossless, so GPU JPEG paths (nvJPEG, torchvision decode_jpeg, DALI JPEG) are out. I found no evidence of a GPU PNG decoder in nvImageCodec, so PNG decode would fall back to CPU. Pre-decoding once into a raw cache (already done) removes decode from the hot path entirely, so the remaining question is I/O and host overhead, not codec speed.

### Cited Findings
- nvImageCodec's install page lists codec modules nvjpeg, nvjpeg2k and nvtiff and describes "automatic fallback" to third-party codecs, but does not mention PNG or WebP at all — [nvImageCodec docs](https://docs.nvidia.com/cuda/nvimagecodec/index.html)
- A benchmark write-up states some formats such as PNG and WebP fall back to CPU decoding in nvImageCodec (secondary source, a blog) — [Arik Poz, nvImageCodec vs OpenCV](https://arikpoz.github.io/posts/2025-04-07-fast-image-loading-with-nvidia-nvimagecodec)
- No GPU decoder for PNG is listed in the nvImageCodec extension list on GitHub (search summary; I did not read the full list) — [NVIDIA/nvImageCodec](https://www.github.com/NVIDIA/nvImageCodec)
- The L3 paper (research format, not NVIDIA stack) reports 9.29x higher decode throughput on an A100 vs lossless PNG on an Intel Xeon CPU for Full HD Cityscapes; this shows PNG is slow on CPU but is not a GPU-PNG measurement — [arXiv 2208.08711](https://arxiv.org/pdf/2208.08711.pdf)
- FFCV paper: on the standard PyTorch ImageNet loader, reading alone took ~70 s but adding decode, random crop and flip raised loading to ~1200 s; per-sample processing, not disk, was the bottleneck; idealized headroom up to 30x. FFCV's remedy is a custom file format, caching, async transfer and Numba JIT pipelines (hardware: 8xA100, 96 vCPU, so not comparable to 4 cores) — [FFCV, arXiv 2306.12517](https://arxiv.org/html/2306.12517v1)
- QOI vs others: QOIR benchmark table normalizes decode speed to QOIR = 1.0, QOI 0.700, WebP lossless 0.325 (relative, not MB/s; PNG-family row is ZPNG, not libpng) — [nigeltao/qoir](https://github.com/nigeltao/qoir)
- QOI README claims 3-4x faster decoding than stb_image (not libpng/Pillow) — [phoboslab QOI](https://phoboslab.org/log/2021/11/qoi-fast-lossless-image-compression)
- One Zig QOI port reports ~1128 MB/s average decompression on a Ryzen 7 3700U (single machine, single codec) — [ikskuh/zig-qoi](https://github.com/ikskuh/zig-qoi)
- Google's WebP lossless study says WebP lossless decode speed is similar to PNG; a user on the WebP list reported WebP reading ~2x slower than PNG on their workload — [Google WebP study](https://developers.google.com/speed/webp/docs/webp_lossless_alpha_study); [webp-discuss thread](https://groups.google.com/a/webmproject.org/g/webp-discuss/c/FPOfZs2cCS4)

### Inferences
- Required consumption rate is arithmetic, not measured: if a T4 does ~3-6 it/s at batch 32, a single diffusion trainer needs ~100-190 samples/s; at 2 trainers ~200-380 samples/s total. Codec at 256px batch 8 at maybe 2-4 it/s needs ~16-32 samples/s per trainer. The user's own figure of 40-100 ms/PNG decode gives ~10-25 samples/s per core, so ~40-100 samples/s on 4 cores at best, shared: the original PNG pipeline is plausibly 2-4x too slow for diffusion, which is consistent with data-bound runs. (Inference from the user's figures.)
- Any compressed-lossless option (QOI, WebP lossless, PNG level 1) decodes a whole 2040x1360 image (8.3 MB raw) to get one 128x128 crop (49 KB), a ~170x waste, unless images are tiled/chunked. Even at ~1 GB/s QOI decode that is ~8 ms/sample per core, i.e. ~120 samples/s/core at best, so compressed formats are only competitive if stored as tiles, and then random crops straddle tile boundaries. Raw uint8 memmap remains the lowest-CPU option. (Inference; QOI MB/s is from a single laptop CPU.)
- GPU decode (DALI, nvImageCodec, torchvision decode_jpeg(device='cuda')) is JPEG/JPEG2000-centred and therefore either lossy (violates exact pixels) or has no GPU PNG path; not recommended here.
- Pillow-SIMD and TurboJPEG are JPEG/resize accelerators; they do not speed up PNG (zlib inflate) decode. I did not find a source quantifying this; treat as inference from the format.

### Gaps
- No measured PNG decode samples/s on 4 Kaggle cores for libpng/Pillow/spng/OpenCV.
- No measured T4 it/s for the user's actual U-Net/codec (needs own profiling).
- No NVIDIA primary statement that PNG is unsupported on GPU in DALI/nvImageCodec; only absence from docs and one blog.
- Pillow-SIMD, WebDataset and TurboJPEG were not researched beyond reasoning (budget).

## 2. Memmap raw cache behaviour, compressed alternatives, GPU-resident cache, DataLoader tuning

### Takeaway
A 41 GB memmap on a ~29 GB RAM box will not fully fit in page cache, so a fraction of random crop reads will hit disk; reports show throughput can collapse 20x when the file exceeds RAM with random access. A per-crop read is ~128-256 scattered row reads, so latency-bound IOPS on Kaggle's disk matters more than bandwidth; this needs a measurement on the actual disk.

### Cited Findings
- PyTorch forum: with a float32 memmap larger than RAM and shuffled access, throughput fell from ~60 batches/s (fits in RAM) to ~3 batches/s, with ~500 MB/s constant disk reads versus ~8 MB/s needed, using 20 workers; reading in order was fast; the thread did not reach a confirmed fix for random access — [PyTorch forum](https://discuss.pytorch.org/t/numpy-memmap-throttles-with-dataloader-when-available-ram-less-than-file-size/83274)
- Same thread: create the memmap once in `__init__`/lazily per worker rather than in every `__getitem__` — [PyTorch forum](https://discuss.pytorch.org/t/numpy-memmap-throttles-with-dataloader-when-available-ram-less-than-file-size/83274)
- Another forum thread reports DataLoader worker RSS appearing to equal the full memmap size (shared page-cache pages counted per process) — [PyTorch forum](https://discuss.pytorch.org/t/dataloader-worker-ram-at-full-dataset-size-with-mem-map-npy/165601)
- A direct-I/O project claims better random-read speed than np.memmap on a 54 GB file, and says OS readahead on random access fetches pages that are never used (single-author benchmark) — [jtc1246/direct-mmap](https://github.com/jtc1246/direct-mmap)
- PyTorch tuning guide: default num_workers=0 is synchronous in the main process; tune num_workers per workload; use pin_memory with non_blocking copies for async host-to-device transfer — [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- Course material warns against more workers than cores (e.g. 64 workers on 16 cores): context switching and memory overhead; recommends persistent_workers=True to avoid worker respawn (secondary) — [TU Berlin data loading chapter](https://commit-web.commit.tu-berlin.de/en/book/sci-python/chapter/ch13/section/s04)

### Inferences
- Layout matters: with an (H, W, 3) per-image layout, a 128x128 crop is 128 separate 384-byte segments at 6120-byte row stride, each pulling at least one 4 KB page (plus readahead), i.e. ~0.5 MB of page-granular reads per sample (arithmetic). For 256x256: 256 segments of 768 B. If the page cache holds ~70% of the file the miss rate is ~30% for uniform random crops, so ~40-80 disk page reads per sample. Whether Kaggle's disk handles this at ~200-400 samples/s (about 10-30k small reads/s) is unmeasured and is the key risk.
- Mitigations that do not change the distribution: (a) madvise/posix_fadvise RANDOM to disable readahead (np.memmap does not expose this directly; use mmap module + `madvise(mmap.MADV_RANDOM)` on Python 3.8+); (b) store images in a tiled layout (e.g. 256x256 tiles contiguous) so a crop touches 1-4 contiguous chunks instead of 128-256 scattered rows, still sampling crop positions uniformly since only the storage layout changes; (c) store as channel-last tiles to keep reads large; (d) shrink the cache footprint (see below).
- Sharing: two trainers on the same machine share the same page cache for the same file, so a single file read by both is better than two copies. Use /dev/shm only for the part that fits: /dev/shm counts against RAM, so putting 41 GB there is impossible with 29 GB RAM; it would only help for a subset (inference).
- Keeping a uniform sampling distribution while caching only part of the data in RAM is not possible without bias, unless the uncached portion is still read from disk. A GPU-resident cache of ~15 GB T4 memory cannot hold 41 GB; a subset-only GPU cache would change the sampling distribution (do not do it) unless it holds everything. Two T4s have 2x15 GB = ~30 GB, still < 41 GB (T4 VRAM 16 GB per card; from general knowledge, not fetched).
- Lossless compression of the cache (lz4/zstd chunks over tiles) could bring 41 GB under ~29 GB if photos compress ~1.5-2x, which is plausible for lossless photos (PNG-level ratios on camera photos are often ~1.5-2.5x, general knowledge, not cited), but then decode of a tile costs CPU. lz4 decodes at multi-GB/s per core (general knowledge, not cited), so a tile-based lz4/zstd layout is the most promising "fits in RAM" option; needs a benchmark.
- Worker tuning guess: with 4 cores shared by 2 trainers, ~2 workers each (so total workers = cores minus the two main processes' needs) rather than 4 each; prefetch_factor 4-8; persistent_workers=True; pin_memory=True. Cheap memmap reads are mostly I/O-wait, so slightly more workers than cores can help when disk-bound, contradicting the generic guidance. Needs to be benchmarked (no source).

### Gaps
- No measurements of Kaggle disk random-read IOPS or throughput for /kaggle/temp or /kaggle/working found.
- No source for page-cache eviction policy specifics or for Kaggle's container memory cgroup counting page cache (cgroup limits can include page cache, which can shrink effective cache below 29 GB; unverified).
- No benchmark of lz4/zstd tile decode vs raw memmap for this crop pattern.

## 3. GPU-side crop and flip from pinned uint8 tensors

### Takeaway
Yes, in principle: crop + flip are pure indexing, so doing them on GPU gives bit-identical pixels and the same distribution if the same index/flip random draws are used. But the host still has to supply source data; GPU cropping only helps if full images (or large windows) reside on GPU or are transferred, which costs far more bytes than the crop. I found no source specifically evaluating this.

### Cited Findings
- PyTorch tuning guide recommends pinned memory plus non_blocking copies for async H2D transfer — [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- The pinned-memory caveat: it is page-locked, cannot be swapped, reduces available RAM (secondary) — [Towards Data Science loop efficiency](https://towardsdatascience.com/improve-efficiency-of-your-pytorch-training-loop/)

### Inferences
- Crop on CPU from memmap is just a strided copy (~49 KB for 128px), negligible CPU work; the cost is page faults and I/O, not the crop arithmetic. GPU cropping removes almost nothing of that and requires moving whole 8.3 MB images, so it is worse unless the whole dataset fits on GPU (it does not at 41 GB vs 2x16 GB).
- The flip is trivial (`np.flip`/torch.flip on a batch). A cheap, distribution-identical improvement: do flip on GPU after transfer, and do uint8->float conversion and normalization on GPU (transfer 4x fewer bytes than float32).
- Batch transfer size: 32x3x128x128 uint8 = 1.5 MB, 8x3x256x256 = 1.5 MB. At PCIe ~10+ GB/s (general knowledge), that is ~0.15 ms, so H2D is not a bottleneck here; the issue is per-step Python and sync overhead.
- Identical distribution means: same RNG-to-(image, y, x, flip) mapping law; drawing positions uniformly in [0, H-h] x [0, W-w] and image uniformly. Moving the draw to GPU RNG changes the exact stream but not the distribution.

### Gaps
- No source quantifying GPU crop speed-ups for this regime.

## 4. Kaggle specifics

### Takeaway
Sources agree on 4 vCPU and ~29 GB RAM for GPU notebooks, ~20 GB persisted output in /kaggle/working, 12 h sessions for GPU/CPU (9 h TPU), and ~100 GB private dataset quota; but /kaggle/temp size and disk throughput are undocumented in what I found. The 41 GB cache cannot be saved to /kaggle/working output (20 GB), so it must be rebuilt or fetched each session, or attached as a dataset.

### Cited Findings
- 4 CPU cores and 29 GB RAM on GPU notebooks (previously 2 cores / 13 GB) — [Medium, Kaggle's new 29 GB RAM GPUs](https://medium.com/@fareedkhandev/kaggles-new-29gb-ram-gpus-the-power-you-need-absolutely-free-b458c3c501ba) (undated secondary); [Packt Kaggle Book](https://packtpub.com/en-mx/product/the-kaggle-book-9781801817479/chapter/working-and-learning-with-kaggle-notebooks-4/section/working-and-learning-with-kaggle-notebooks-4)
- /kaggle/working about 20 GB (older sources said 5 GB); an additional scratch area outside /kaggle/working that is not saved after the session; one older source listed 16 GB scratch — [Packt Kaggle Book](https://packtpub.com/en-mx/product/the-kaggle-book-9781801817479/chapter/working-and-learning-with-kaggle-notebooks-4/section/working-and-learning-with-kaggle-notebooks-4); [Medium, using Kaggle](https://medium.com/data-science/using-kaggle-for-your-data-science-work-a2e78d692395)
- A setup guide lists 73 GB disk available for dual T4 sessions (unverified third-party) — [llcuda Kaggle setup](https://llcuda.github.io/guides/kaggle-setup)
- Session limits: 12 h CPU/GPU, 9 h TPU — [Packt Kaggle Book](https://packtpub.com/en-mx/product/the-kaggle-book-9781801817479/chapter/working-and-learning-with-kaggle-notebooks-4/section/working-and-learning-with-kaggle-notebooks-4)
- Kaggle's RAPIDS guide confirms T4 x2 is a selectable accelerator — [NVIDIA RAPIDS on Kaggle](https://docs.nvidia.com/datascience/deployment/latest/platforms/kaggle/)
- Private dataset limits: Packt reports 100 GB per private dataset and 100 GB total (and "over 100 GB" in a later edition); another Packt title says 200 GB; size is counted uncompressed; no limit on public datasets per Kaggle staff — [Packt, Setting up a dataset](https://subscription.packtpub.com/book/data/9781801817479/3/ch03lvl1sec07/setting-up-a-dataset); [Kaggle forum 116079](https://kaggle.com/discussions/product-feedback/116079)
- Hugging Face private storage quotas exist and can produce 403 errors even when quota appears available — [HF storage limits](https://huggingface.co/docs/hub/en/storage-limits.md); [HF forum](https://discuss.huggingface.co/t/403-error-private-repository-storage-limit-reached-quota-shows-space-remaining/153121)

### Inferences
- Persisting the cache: attach the original PNG dataset (the source, probably much smaller than 41 GB) and rebuild the raw cache at session start; rebuild cost is one full PNG decode of 4,900 images: at 40-100 ms each across 4 cores that is ~1-2 minutes of pure decode (arithmetic from user's figures: 4,900 x 0.04-0.1 s / 4 ≈ 50-120 s) plus ~41 GB of disk writes, so rebuild is likely cheaper than restoring 41 GB over a network, and avoids any dataset size quota. Disk write speed on Kaggle is unmeasured.
- Restore time for 23-41 GB from a dataset or HF at e.g. 100-200 MB/s would be ~2-7 min per 41 GB (arithmetic from assumed speeds; actual Kaggle dataset mount and HF download speeds were not found). Kaggle input datasets are mounted read-only and may be a network-backed mount, so random memmap reads directly from /kaggle/input are risky; copy to local disk first (unverified).
- A lossless-compressed tile cache (see section 2) would shrink persisted size to ~20 GB-ish so it could fit under the 20 GB working output or the dataset quota.

### Gaps
- Official Kaggle docs not retrievable; exact /kaggle/temp size, total disk, disk throughput/IOPS, and whether the memory cgroup counts page cache unverified.
- No measured Kaggle dataset download or Hugging Face download speeds into a Kaggle notebook.
- Whether /dev/shm size is limited on Kaggle (default Docker /dev/shm is 64 MB unless set) not found.

## 5. Host-to-device transfer, Python overhead, CUDA graphs, and diagnosing data-bound runs

### Takeaway
At 1.5 MB per batch, transfer cost is negligible; GPU idle gaps come from worker latency and per-step Python/launch overhead. Diagnose with simple wait-time timing around `next(loader)` plus nvidia-smi dmon.

### Cited Findings
- PyTorch tuning guide: pin_memory plus non_blocking for async copy; num_workers should be benchmarked — [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- Forum thread on CPU-GPU synchronization: `.to('cuda', non_blocking=True)` is truly async only from pinned memory — [PyTorch forum](https://discuss.pytorch.org/t/performance-optimization-re-cpu-gpu-synchronization/157251)
- FFCV demonstrates async transfer and pre-loading as part of removing the data bottleneck — [FFCV](https://arxiv.org/html/2306.12517v1)

### Inferences (from general PyTorch knowledge; not cited from fetched sources)
- Simple data-bound test: time `t_wait` (time blocked in `next(iter)`) vs `t_step` with `torch.cuda.synchronize()` at the boundaries; if wait is >5-10% of step, data-bound. Cross-check by feeding one fixed cached batch repeatedly: if it/s jumps, the loader is the bottleneck.
- `nvidia-smi dmon -s u` shows sm% sampled per second; utilisation sawtooth/below ~90% hints at stalls; `torch.profiler` with a schedule shows gaps between kernels and DataLoader time. `nvidia-smi` util is coarse (any kernel active in the sample interval counts).
- CUDA graphs (torch.compile mode="reduce-overhead") cut launch overhead for small, static-shape workloads; benefit is larger when kernels are tiny or CPU is slow (4 shared cores qualifies). They require static shapes and careful handling of RNG/optimizer; whether they help depends on the model and is untested here. Two trainers sharing 4 cores also compete for Python launch threads, so reducing launch overhead may matter more than data.
- Set `torch.set_num_threads` low in workers (e.g. 1) to avoid oversubscription with 2 trainers.

### Gaps
- No cited benchmarks of CUDA graph speedups for U-Net at batch 32 on T4; no cited dmon documentation (not fetched).

## 6. Verifying the pipeline change keeps the sample distribution identical

### Takeaway
Two-level check: (1) exact-tensor equality for deterministic parts, (2) statistical tests for stochastic parts. Since the cache is a lossless raw copy of the decoded PNG, the pixel content can be checked exactly.

### Cited Findings
- No source found on verifying data-pipeline equivalence; this section is purely my own reasoning.

### Inferences
- Exact-pixel test: for each of the 4,900 images, compare memmap array vs `np.asarray(PIL.Image.open(path).convert('RGB'))` with `np.array_equal` (or a hash such as xxhash of both), including shape and channel order; fails loudly on any difference (EXIF orientation, palette/16-bit/alpha PNGs are the usual traps).
- Exact-sample test: feed the same (image index, y, x, flip) tuples to the old and new loaders; outputs must be bit-identical. Easiest by factoring randomness out of `__getitem__` into an explicit sampler seeded with `torch.Generator`/`numpy.random.Generator(seed)` and replaying it.
- Distribution test: draw ~10^5-10^6 (index, y, x, flip) tuples from the new pipeline (log them, not the pixels) and chi-square/KS test against the intended law (index uniform over images, or over what the original did; y, x uniform on valid range; flip Bernoulli 0.5). Check also that per-worker seeding does not duplicate streams (a classic DataLoader bug: forked workers with identical numpy RNG state).
- Check that image selection weighting is unchanged (uniform over images vs over pixels) and that the crop range includes the last valid offset (off-by-one at `H-h`).
- Compare also the first moments of a pixel statistics sample (mean/std per channel) as a coarse sanity check; not a proof.

### Gaps
- None cited; these are recommendations, not sourced findings.

## Overall recommendation (inference, ordered)
1. Keep raw uint8 cache; open memmap lazily per worker, once; use MADV_RANDOM; benchmark crops/s cold vs warm on the actual Kaggle disk before anything else.
2. If cold reads are too slow, restructure to a tiled layout and/or lz4/zstd tiles so the working set fits in ~29 GB page cache shared by both trainers; sampling positions remain uniform if tiles are read and re-cropped to the sampled window.
3. Tune workers (about 2 per trainer), prefetch_factor, persistent_workers, pin_memory; do flip, float cast and normalization on GPU; consider CUDA graphs only after confirming the GPU is idle due to Python overhead.
4. Rebuild the cache each session from the attached PNG dataset (est. 1-2 min decode, plus write time) rather than restoring 41 GB.
5. Avoid GPU image decode, lossy formats, and subset caches (all risk exact-pixel or distribution changes, or lack PNG support).
