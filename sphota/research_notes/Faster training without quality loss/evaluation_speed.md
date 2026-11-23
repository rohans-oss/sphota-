# Faster evaluation / decoding on NVIDIA T4 without changing results (as of Nov 2026)

Scope note: about 8 tool calls were used. Almost no published T4 benchmark exists for a small pixel-space UNet2DModel; most evidence is for Stable-Diffusion-class latent UNets. Items without a source are marked as inferences or gaps, and all of them need local measurement.

## 1. Real speedups on T4: fp16 autocast, batched CFG, channels_last, torch.compile, SDPA, CUDA graphs

### Takeaway
No source gives measured T4 numbers for this model. The best available evidence is that fp16 plus PyTorch 2 SDPA gives roughly 20-28% over vanilla attention on non-Ada GPUs, and compile adds only a few percent to about 15% on T4 depending on batch size. Compile risks (recompiles per image size, large-batch regression) are documented. The large wins for this workload are likely fp16 tensor cores and batch-2 CFG, but these are inferences that must be measured.

### Cited Findings
- Diffusers docs: PyTorch loads fp32 by default, and changing dtype is the simple first speedup. bf16 is more robust, fp16 "may be more prone to numerical errors". SDPA is on by default with PyTorch >= 2.0 and auto-selects flash, xFormers-style efficient, or native C++ backends — [Diffusers fp16 docs](https://huggingface.co/docs/diffusers/optimization/fp16)
- Diffusers docs: compile the UNet with `channels_last`, `fullgraph=True`; the first call is slow; a different image size "retriggers compilation"; `dynamic=True` (and `use_duck_shape=False`) can avoid recompiles but "not all models may benefit out of the box"; `mode="max-autotune"` uses CUDA graphs; keep scheduler sigmas on CPU to avoid GPU syncs (stronger after compile) — [Diffusers fp16 docs](https://huggingface.co/docs/diffusers/optimization/fp16)
- Diffusers docs: regional compilation (`compile_repeated_blocks`) cuts compile time 8-10x on repeated-block models; it targets transformer-style repeated blocks, and a UNet2DModel would need `_repeated_blocks` defined — [Diffusers fp16 docs](https://huggingface.co/docs/diffusers/optimization/fp16)
- PyTorch blog (Stable Diffusion, fp32): out-of-the-box PyTorch 2 SDPA vs xFormers gave "moderate speedups of up to 2%" on datacenter cards such as A100 or T4; compile added a typical 2-3% before graph-break fixes — [PyTorch blog](https://pytorch.org/blog/accelerated-diffusers-pt-20/)
- Same blog, fp16: SDPA gave 20-28% over standard attention across the tested GPUs (T4 not listed individually in text), on par with xFormers; after graph-break fixes, compile gave about 13-22% for most workloads. The blog states a performance decrease on T4 at batch size 16 due to memory pressure — [PyTorch blog](https://pytorch.org/blog/accelerated-diffusers-pt-20/)
- Diffusers torch2.0 docs table (as read through a search summary, column labels inferred, so treat as indicative): T4 fp16, compile+SDPA vs xFormers about 8.4% at batch 4 and 3.3% at batch 8; PyTorch blog figures about 2% (batch 1), 10.5% (batch 2), 14% (batch 4) — [Diffusers 0.14 torch2.0 docs](https://huggingface.co/docs/diffusers/v0.14.0/en/optimization/torch2.0), [PyTorch blog](https://pytorch.org/blog/accelerated-diffusers-pt-20/)
- PyTorch/Diffusers blog 2025: compiling the DiT gave about 1.5x on H100 (not T4). Cold compile took 67 s for the full model, 9.6 s regional, 2.4 s with a warm inductor cache. `dynamic=True` is advised for multiple resolutions (and `mark_dynamic` is the PyTorch-recommended way) — [PyTorch blog 2025](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
- SDPA on Turing: flash backend does not run on sm75/older; the memory-efficient backend can be forced with `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`. Forcing an unsupported backend raises an error. Evidence is from a gist and forum posts, not official docs — [gist](https://gist.github.com/am009/be749ce4b6133a2208366205e9d9728b), [PyTorch forum](https://discuss.pytorch.org/t/troubles-with-scaled-dot-product-attention/182366)
- NVIDIA: for FP16 Tensor Cores choose channel counts divisible by 8; NHWC is supported in PyTorch via channels_last — [NVIDIA conv guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional)
- fast.ai: channels_last works on Volta/Turing+, but unsupported operators can cause repeated format conversions ("channel thrashing") — [fast.ai docs](https://docs.fast.ai/callback.channelslast.html.md)

### Inferences
- The model channels (64/128/256/256) are all multiples of 8, so fp16 conv should hit Tensor Cores. T4 fp16 tensor-core throughput is far higher than its fp32 CUDA-core throughput (from general knowledge, not fetched here), so fp16 autocast is the biggest lever, likely much larger than the 1.2x seen in SD attention-bound benchmarks. This is unverified for this model and must be measured.
- Attention at 1/4 resolution for 768x512 is 192x128 = 24,576 tokens per image; vanilla attention would build a 24.6k x 24.6k matrix (about 1.2 GB per head in fp16). Memory-efficient SDPA matters for memory and speed. Check that diffusers' Attention in UNet2DModel actually dispatches to SDPA (`AttnProcessor2_0`). This is an inference.
- Batch-2 CFG (cond and uncond concatenated) should help most at batch 1 where kernels under-fill the GPU. At 768x512 the GPU is probably already saturated, so expect modest gain. Note torch.compile gains grew with batch size 1->4 in the T4 table, which supports "batching helps a bit". Numerics: batch-2 changes cuDNN algorithm choice, so results differ at noise level only.
- Compile is a poor fit if image sizes vary a lot (recompile per shape, each possibly tens of seconds to minutes; T4 numbers are unknown). Use it only if images are bucketed or padded to few shapes, or with `dynamic=True` after testing. With `fullgraph=True` fix graph breaks first. Gain is a few percent to about 15% at best, so the priority is below fp16 and batching.
- CUDA graphs (via compile `reduce-overhead`, or manual capture) only help if the GPU is launch-bound. For batch 1 at 768x512 with a 4-level UNet it likely is not. They also require static shapes.
- `torch.inference_mode()` or `no_grad` is exact (no numerical change); gain is small and mostly memory. Guidance steps need grad through g_a only, so wrap the UNet call in `inference_mode` or `no_grad` and keep g_a outside it.
- xFormers is not needed; PyTorch SDPA with the efficient backend is on par (PyTorch blog above).

### Gaps
- No T4 benchmark of fp32 vs fp16 autocast for a UNet2DModel of this size, none for pixel-space 768x512, and none for channels_last on T4 specifically.
- No sourced compile-time numbers on T4 or for varying image sizes.
- Whether bf16 is an option: T4 lacks native bf16 tensor cores (general knowledge, not sourced here), so fp16 is the only choice. The model was trained under fp16 autocast, so fp16 inference matches training numerics; this is a point in its favour but not proof.

## 2. Reusing computation across sweeps (CFG scales, caches, reorderings)

### Takeaway
I found no source showing exact sharing of eps predictions across CFG scales; by the trajectory-divergence argument, none exists except at the first step. The safe savings are caching everything that does not depend on x_t.

### Cited Findings
- No cited research found on exact reuse across CFG scales. (An absence of evidence, not a proof.)
- clean-fid: custom reference statistics are built once with `fid.make_custom_stats`, stored in a local cache, and can be checked via `fid.test_stats_exists` — [clean-fid](https://github.com/GaParmar/clean-fid)
- EasyOCR: create `Reader` once and reuse; the first GPU call is slow (exclude it from timing) — [EasyOCR search results via ray forum and docs](https://github.com/JaidedAI/EasyOCR/issues/358)

### Inferences
- Step 1 sharing: at the first step (same x_T per seed), the cond and uncond eps are identical across all 5 CFG scales, so one pair of passes serves all scales (saves 1 of 25-50 steps, about 2-4%). Only the combination eps_u + w(eps_c - eps_u) differs. After that, x_t diverges. Scale w=0 or w=1 special cases: w=1 means the unconditional pass is unnecessary (eps = eps_c) if the code uses that CFG form, and w=0 skips the conditional pass. Check the CFG formulation before using this.
- Guided vs unguided runs with the same seed share nothing after step 1 once guidance modifies x_t. Unguided (no bin-consistency) runs at the same CFG scale sharing a trajectory with guided ones is impossible after the first guided step, but guidance starts only at t<=700 and runs before that are identical. Reuse the shared prefix: for steps with t>700, the guided and unguided trajectories are identical (if guidance only acts at t<=700), so cache x_t at the first guided step. This is exact and could save 30-50% of the compute of a paired guided/unguided run. Verify that the schedule and RNG are consistent (eta=0 means no noise injection, so this should hold exactly).
- Same prefix argument applies across step-count settings only if the timesteps coincide, which they do not for 25 vs 50 in general, so it does not apply.
- Per-image caches (exact): codec compress/decompress bitstreams and decoded latent y_hat per image and bitrate (independent of the sampler seed/CFG scale); text/conditioning-free inputs; the original image's OCR output, FaceNet embedding, LPIPS-AlexNet features; clean-fid reference statistics (once per dataset). Load LPIPS/EasyOCR/FaceNet models once per process.

### Gaps
- No literature found on CFG-sweep trajectory sharing; the prefix-sharing idea is my own reasoning and must be verified numerically (bitwise equality of x_t at the branch step).

## 3. Faster bin-consistency guidance through g_a (fp16, channels_last, autocast safety)

### Takeaway
There is no source on autocast safety for rounding-bin consistency losses. It is the highest-risk change in this project; the 0.5-bin granularity and a GDN denominator both argue for keeping g_a in fp32, at least for the final consistency-rate computation.

### Cited Findings
- Diffusers docs: fp16 "may be more prone to numerical errors" than bf16 — [Diffusers fp16 docs](https://huggingface.co/docs/diffusers/optimization/fp16)
- NVIDIA: FP16 Tensor Core conv wants channel counts divisible by 8; channels_last is supported but unsupported ops can cause format thrashing — [NVIDIA conv guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional), [fast.ai](https://docs.fast.ai/callback.channelslast.html.md)

### Inferences
- fp16 has about 3 decimal digits (10-bit mantissa). Latent values of magnitude ~10-30 have fp16 spacing of about 0.0078-0.0156, so rounding error is small relative to 0.5 bins, but values close to a bin boundary (within about 0.01) can flip. That moves the consistency rate by a fraction of the boundary-near mass, not necessarily noise-free. GDN involves sqrt/division of squared activations, which is a known source of fp16 range trouble; autocast would normally keep some ops in fp32, but custom GDN layers may run in fp16.
- A safe hybrid: run the 3 gradient steps with g_a under autocast (gradient direction needs only approximate accuracy), but always compute the reported consistency-rate and the final bin decisions in fp32. Parity-test that gradients (cosine similarity vs fp32) and final outputs agree. Use the same parity protocol as section 5. Do not take this as sourced; no published test exists.
- Cheaper exact alternatives for guidance: use `torch.autograd.grad` on the input only (not parameters; set `requires_grad_(False)` for g_a weights, saving weight-gradient convs); use `no_grad` for the U-Net; reuse the same g_a forward for the 3 steps only if the input does not change (it does, so no). channels_last with tf32 is moot on T4 (no TF32 on Turing; general knowledge).
- Gradient steps cost: 3 forward+backward through a 4-layer conv net on the full image versus 2 UNet forwards per step; probably under 30% of the step cost, so guidance is not the dominant term. Profile first.

### Gaps
- No empirical data on autocast with GDN layers or on rounding-based loss parity.

## 4. Faster metrics pipelines (EasyOCR, FaceNet, LPIPS, clean-fid) and 2-GPU data parallel

### Takeaway
Evidence here is thin. The documented points: EasyOCR `batch_size` >1 is faster at a memory cost, confirm GPU use, reuse Reader. I found no authoritative sources for FaceNet/LPIPS speed or for multi-GPU splitting.

### Cited Findings
- EasyOCR README: `batch_size>1` makes it faster but uses more memory; the Reader loads models once; first prediction is slow; silent CPU fallback is possible and can be several times slower (check `nvidia-smi`) — [EasyOCR (qhduan mirror README)](https://github.com/qhduan/EasyOCR), [PyImageSearch](https://pyimagesearch.com/2022/03/14/improving-text-detection-speed-with-opencv-and-gpus/), [imagetotable blog (secondary)](https://imagetotable.ai/blog/ocr-too-slow-speed-optimization)
- A forum user reported about 30 minutes for 1,000 images in batches of 8 with EasyOCR on GPU (no confirmed fix) — [Ray forum](https://discuss.ray.io/t/increase-efficiency-using-pytorch-gpu-for-inference/6858)
- clean-fid: custom stats cached locally; the README fetched here gave no info on batch size, workers or GPU — [clean-fid](https://github.com/GaParmar/clean-fid)

### Inferences
- Split into stages: decode all images first and save PNGs (lossless), then run metrics in separate passes with each model loaded once. This avoids model reload and GPU contention.
- EasyOCR: cache the original-image OCR results once; pass `batch_size` and `gpu=True`; only crop-level OCR on text regions if that is the protocol (do not change protocol). EasyOCR has CPU-bound detection post-processing (OpenCV), so run it in a CPU process pool or overlapped with GPU decode. This is an inference, not sourced.
- Multi-GPU on 2 T4s: the images are independent, so run 2 processes with `CUDA_VISIBLE_DEVICES=0/1`, each decoding a disjoint half of (image, seed, config) tasks. Results are exact (same per-image numerics; only scheduling differs) and nearly 2x, provided CPU cores and disk I/O keep up and each task is seeded by its own explicit seed, not by a shared sequential RNG. This is standard practice, not backed by a specific source here. Do not use DataParallel (it splits batch; at batch 1 it does nothing).
- Metrics (MS-SSIM, PSNR, LPIPS) on 768x512 images are cheap relative to diffusion; the 20-30 h is dominated by sampling.

### Gaps
- Found no sourced facts on facenet-pytorch speed, LPIPS fp16, or clean-fid throughput tuning. Not verified whether fp16 is safe for FaceNet embeddings (cosine-similarity thresholds could flip near the boundary).

## 5. Parity protocol for trusting fp16 / batched decoding

### Takeaway
No published protocol was found for this; the following is a proposed design built from standard paired-comparison practice and the risk analysis above. Treat it as a design proposal, not as a cited method.

### Cited Findings
- Diffusers docs warn that fp16 is more error-prone than bf16, which motivates a parity check rather than assuming equivalence — [Diffusers fp16 docs](https://huggingface.co/docs/diffusers/optimization/fp16)
- PyTorch's T4 observation that large batches can regress due to memory pressure supports re-timing each configuration — [PyTorch blog](https://pytorch.org/blog/accelerated-diffusers-pt-20/)

### Inferences (proposed protocol)
1. Fix everything else (seed, x_T, scales, steps, schedule). Run reference (fp32, two separate CFG calls) and candidate (fp16 autocast, batched CFG, optional compile) on the same 20 images (choose across sizes and text/face-heavy ones, since TFR/FIR depend on these) and same seeds.
2. Numerical level: record per-step max/mean abs difference of eps and of x_t versus fp32, and final-image PSNR between the two decodes (very high, e.g. well above the codec's own PSNR, means noise level). Also compare the seed-to-seed spread in the reference as the natural noise floor: candidate-vs-reference differences should be much smaller than seed-to-seed differences.
3. Metric level: compute paired differences per image for LPIPS, PSNR, MS-SSIM, TFR/CER, FIR, consistency rate; report mean difference with 95% bootstrap CI and compare against the reference's own seed-to-seed paired difference. Accept if the CI of the mean difference is within a pre-registered tolerance (for example a small fraction of the smallest effect you intend to claim) and no systematic sign bias.
4. Discrete metrics (TFR, FIR, consistency rate) are threshold-based: report the count of flipped decisions out of N and check that flips are balanced in both directions (McNemar-style). Consistency rate is the most fragile; keep its computation fp32 on the decoded latent regardless (section 3).
5. Run a bitwise-determinism check: the same config twice should give identical outputs; if not (nondeterministic cuDNN or atomic ops), the noise floor is non-zero and must be measured by repeat runs of the reference too. `torch.backends.cudnn.benchmark=True` can change algorithm choice between runs.
6. Freeze the final config; run all sweeps with it. Optionally re-verify one sweep point in fp32 as a spot check.

### Gaps
- No sourced tolerances; thresholds must be chosen by the project. The number 20 images is the user's suggestion; its statistical power for rare events (FIR flips) is limited.

## Priority order (my synthesis, unmeasured)
1. Cache all image-level invariants and decode artifacts; split decode and metrics into separate stages; run 2 processes on 2 T4s (exact, near 2x).
2. Reuse the shared prefix of guided/unguided runs (t>700) and the first step across CFG scales; verify bitwise.
3. fp16 autocast for the UNet (matches training), parity-tested; fp32 for consistency-rate computation.
4. Batch-2 CFG; SDPA efficient backend check; `inference_mode`.
5. channels_last (test on T4; may help with fp16 convs). torch.compile only if image sizes are bucketed; expect single-digit to about 15% on T4.
