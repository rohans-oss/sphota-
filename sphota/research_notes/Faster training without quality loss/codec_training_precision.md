# Faster fp32 -> mixed-precision/compiled training of a GMM hyperprior learned image codec on NVIDIA T4

Scope note: web search was low-yield for this niche. I found NO published paper or GitHub issue that reports training a Balle/Minnen/Cheng2020/ELIC/MLIC/TCM model with AMP and reports RD results. Everything on "what is safe in fp16" below is therefore derived from library source code and PyTorch AMP docs, plus first-principles reasoning, and is labelled as inference. It must be validated by the A/B protocol in the last section.

## Q1. Does anyone report AMP (fp16/bf16) training of LIC models and what happened to RD?

### Takeaway
I found no direct evidence either way (no RD numbers for AMP-trained LIC models, no CompressAI issue/discussion about fp16 GDN). The only adjacent LIC literature is post-training mixed-precision quantization for inference, which is a different problem.

### Cited Findings
- Searches for AMP/fp16 training of LIC models returned only post-training/inference mixed-precision quantization work (per-layer bit-width assignment), e.g. [Flexible Mixed Precision Quantization for LIC](https://arxiv.org/abs/2506.01221) and [DynaQuant](https://arxiv.org/abs/2511.07903); these are not training-time AMP.
- CompressAI issue #92 reports predicted bpp randomly becoming NaN in a hyperprior-based codec trained in (apparently) normal precision; the reporter's mitigation was lower learning rate plus gradient-norm clipping, suspecting gradient explosion. The thread does not mention fp16/autocast. — [CompressAI #92](https://github.com/InterDigitalInc/CompressAI/issues/92)
- A paper on streaming/TensorRT deployment of LIC codecs (CompressAI-based) reportedly identifies GDN as a large inference-cost component and proposes a replacement ("1DN") with no accuracy loss; I could not read the PDF text (binary extraction failed) so numbers are unverified. — [arXiv 2208.01641](https://arxiv.org/abs/2208.01641) (abstract confirms only pipelining/memory-model optimisations, not GDN details; the GDN/1DN claim comes from a search-result summary and is UNVERIFIED)
- The Minnen&Singh channel-wise AR paper trains with batch 8, 256x256 crops, and reports smaller patches hurt RD while larger gave little benefit - a patch-size result, not a precision result. — [arXiv 2007.08739](https://arxiv.org/pdf/2007.08739)

### Inferences
- Absence of reports is itself weak evidence: most LIC work trains in fp32 (or TF32 on Ampere+), so there is no community recipe to copy. Treat AMP for this codec as an experiment.

### Gaps
- No RD (BD-rate/PSNR) data for AMP-trained LIC models found. No CompressAI discussion on fp16 GDN found. No bf16 option exists on T4 anyway (no native bf16 tensor cores).

## Q2. What is safe in fp16, what must stay fp32; GDN efficiency; recipe

### Takeaway
Plain convs are the natural fp16 candidates; GDN's normalization, the Gaussian/mixture CDF/likelihood, log2 rate sum, EntropyBottleneck and the 1e-9 likelihood bound should be kept in fp32. CompressAI's GDN is already a 1x1 conv (F.conv2d(x**2, gamma, beta)) with rsqrt; under autocast the conv would run in fp16 and x**2 can overflow/lose precision, so wrap it in fp32.

### Cited Findings
- CompressAI GDN computes `norm = F.conv2d(x**2, gamma, beta)` with gamma reshaped to CxCx1x1, then `rsqrt` (or `sqrt` for IGDN); beta has min 1e-6 via NonNegativeParametrizer; gamma init 0.1*I; no dtype handling or autocast code. — [gdn.py](https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/layers/gdn.py)
- NonNegativeParametrizer: `out = lower_bound(x)**2 - pedestal`, bound=(minimum+reparam_offset^2)^0.5, reparam_offset default 2^-18 (the value is from my memory of CompressAI, not from the fetched text; the fetched code showed the formula only). Parameters are reparametrized every forward call. — [parametrizers.py](https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/ops/parametrizers.py)
- LowerBound is `torch.max(x, bound)` via a custom autograd function; backward passes gradient if `x >= bound` or `grad_output < 0`; `bound` is a float32 buffer created with `torch.Tensor([...])`; no dtype handling. — [bound_ops.py](https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/ops/bound_ops.py)
- GaussianConditional: CDF via `0.5*erfc(-x/sqrt2)` ("maximizes numerical precision"), likelihood = upper - lower, scale lower bound `scale_bound=0.11`, likelihood floor default 1e-9; no autocast code in the file. GaussianMixtureConditional uses scale clamp (0.11, 256) and TINY=1e-10. EntropyBottleneck uses `torch.rand_like(x) - 0.5` noise in training. — [entropy_models.py](https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/entropy_models/entropy_models.py)
- PyTorch autocast: conv1d/2d/3d, linear, matmul etc. cast to fp16; `exp, log, log2, pow, rsqrt, reciprocal, sum, mse_loss, softmax, norm, softplus` autocast to fp32; `erfc` is NOT in either list (so it runs in the input dtype - i.e. fp16 if fed fp16 tensors); addcdiv/addcmul promote to widest. Custom autograd functions need `custom_fwd/custom_bwd` (use `cast_inputs=torch.float32` to force fp32). — [PyTorch AMP docs](https://docs.pytorch.org/docs/2.15/amp.html)
- Generic NaN-under-AMP guidance in PyTorch issues: force the loss/sensitive region to fp32 by locally disabling autocast. — [pytorch #40497](https://github.com/pytorch/pytorch/issues/40497)
- Original mixed-precision recipe: fp32 master weights + loss scaling. — [Micikevicius et al.](https://arxiv.org/abs/1710.03740)

### Inferences
- Likely safe fp16: the stride-2 5x5 convs of g_a/g_s, h_a, h_s (including the 1728-channel output conv), bias adds. Outputs fed to GMM params (means, log-scales, mixture logits) should be `.float()` immediately after h_s (or after the conv). Note rate-critical parameter precision: fp16 has ~3 decimal digits; means/scales with relative error ~1e-3 shift likelihoods slightly -> this is the main quality risk; casting to fp32 afterwards does not recover lost mantissa, so the last h_s conv could also be kept fp32 (it is small in FLOPs only if the input is 16x16 spatially; verify).
- Must be fp32: the whole likelihood path (erfc differences, 0.11 scale clamp, 1e-9 floor, log2 sum), EntropyBottleneck (its internal cumulative logits and the 1e-9 floor), and the MSE term. A 1e-9 floor is below fp16's min subnormal (~6e-8) so it would silently become 0 in fp16 -> log(0) = -inf -> NaN bpp. This is arithmetic fact; the failure mode in practice is not documented for LIC.
- GDN: x**2 overflows fp16 at |x|>256 (max 65504); latents/activations in early layers of a 255-scaled-input-free [0,1] network are usually smaller but untrained/spiky activations can overflow; gamma accumulation over 128 channels adds risk. Safest: run GDN/IGDN in fp32 (`torch.autocast(enabled=False)` + `x.float()`), accept the extra memory traffic; GDN is elementwise/memory-bound anyway so the fp32 cost is moderate. Alternative: run only the 1x1 conv in fp16 with fp32 x**2 - not recommended without testing. GDN with fp16 params is not an option: keep fp32 master params (autocast does this by default).
- Recipe: `torch.autocast('cuda', dtype=torch.float16)` around encoder/decoder convs; custom fp32 islands for GDN + entropy model + loss; `GradScaler` (needed because fp16 gradients can underflow; with lambda*255^2*MSE the loss scale is large already but gradients of bpp through g_a are small); main and aux optimizers both stepped via the scaler - note aux loss gradients should go through `scaler.scale(aux_loss).backward()` and `scaler.step(aux_opt)` separately; since aux_loss depends only on fp32 quantiles parameters, running it unscaled in plain fp32 is simplest and cheaper to reason about (recommendation, untested). Watch for scaler skipping steps: log the number of skipped steps; if >~0.1% investigate, as skipped steps change "same iteration count" semantics slightly.
- A faster GDN equivalent exists only as an approximation (1DN, other replacements) which changes the model - disallowed by constraints. Exact speedups possible: fuse x**2 -> conv -> rsqrt -> multiply with torch.compile (see Q4), or precompute reparametrized gamma once per forward (already once per call).

### Gaps
- No published evidence of GDN stability or instability in fp16 training. No measured RD impact for fp16 convs in LIC. Could not verify `reparam_offset` default or 1DN details from fetched text.

## Q3. Expected end-to-end AMP speedup on T4; compute- vs memory-bound at batch 8

### Takeaway
No LIC-specific T4 measurement found. General T4 FP16-vs-FP32 numbers for CNNs are about 1.8-2.2x for large batches; for this codec the realised end-to-end gain is probably lower (my estimate roughly 1.2-1.8x) because GDN, entropy model and noise/rounding are elementwise memory-bound fp32 work that AMP does not speed up. This is an estimate, not a measurement.

### Cited Findings
- 2026 benchmark paper (DEEP-GAP) reports on T4 FP16 gives approx 1.8-2.2x speedup over FP32 across most batch sizes; ResNet18 about 2.0x, ResNet50/101 about 2.1-2.3x at large batch (per search summary; snippet does not say whether AMP or channels_last was used). — [arXiv 2604.14552](https://arxiv.org/pdf/2604.14552)
- PyTorch's AMP post reports 1.5x-5.5x over FP32 on V100 across models (V100, not T4). — [PyTorch blog](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
- Forum thread: for a single conv, fp16 was slower in default layout but channels_last avoided permutes (0.00149 s -> 0.00084 s); another user saw slowdowns with channels_last+AMP. — [PyTorch forums](https://discuss.pytorch.org/t/2x-faster-training-than-default-precision-when-model-is-just-3-linear-layer-but-when-the-model-is-just-1-conv-layer-mixed-precision-training-is-slower-than-default-precision/131291)
- PyTorch issue on T4: cuDNN may pick non-tensor-core algorithms when its heuristics think they are faster for particular conv shapes. — [pytorch #42311](https://github.com/pytorch/pytorch/issues/42311)

### Inferences
- My back-of-envelope (not sourced): g_a layers are about 0.16+1.68+0.42+0.16 = about 2.4 GMAC/img (5x5, N=128, 256x256 crop), g_s similar, so about 5 GMAC/img fwd, about 30 GFLOP/img fwd+bwd, about 240 GFLOP/iter at batch 8. At realistic T4 fp32 throughput (~4-6 TFLOPS sustained, T4 is 70 W power-limited) that is roughly 40-60 ms/iter of conv math, i.e. the conv part is compute-bound, not launch-bound, at 256x256 batch 8. The heaviest layers (128->128 5x5 at 64x64 etc.) are large enough for tensor cores to help. Elementwise: GDN at 128x128x128x8 fp32 = 67 MB tensors per op, several passes -> memory-bound (T4 ~320 GB/s), likely 10-25% of step time. The h_s 1728-channel output at 16x16 and the GMM likelihood (3 components x 192 x 16x16 x 8) are small-ish but have many elementwise ops (erfc x6, differences, mixture sum, log).
- Because of Amdahl on the fp32 islands, plan on an unknown fraction; profile first with `torch.profiler` (cuda_time per op) to measure conv vs elementwise share before deciding.
- Channels_last with fp16 convs is generally recommended for tensor cores on Turing (needs channel counts multiple of 8: 3 input channels of the first conv is awkward - first conv may be slow/ not tensor-core eligible; 128/192/1728 are fine). Not benchmarked here for this model.

### Gaps
- No T4 measurement for LIC/conv autoencoders of this size found. Compute- vs memory-bound split is inferred, needs profiling.

## Q4. torch.compile, CUDA graphs, channels_last, fused Adam on T4; known bugs with CompressAI

### Takeaway
Evidence is thin. torch.compile gains for small CNNs on T4 look modest (about 9% in one HF ResNet batch-4 reduce-overhead row; units unstated); at batch 8 / 256x256 launch overhead is not the bottleneck, so CUDA graphs are unlikely to matter much. One known CompressAI compile issue (EntropyBottleneck noise) was fixed in 2026.

### Cited Findings
- CompressAI PR #350 (reported merged Mar 2026) replaces in-place `uniform_` with `torch.rand_like` in EntropyBottleneck.quantize because it caused graph breaks/compilation failures with `torch.compile(fullgraph=True)` (aten.uniform); the author could not give a minimal repro; eager vs compiled outputs differ under same seed due to RNG handling. — [PR #350 mirror](https://gh.r-mj.com/InterDigitalInc/CompressAI/pull/350) (mirror page not directly fetchable by me; claim from search summary). Note the current master entropy_models.py I fetched already uses `torch.rand_like`.
- No CompressAI issue/PR found about GDN + torch.compile. The `LowerBound` custom autograd function with `torch.jit.is_scripting` branching and CompressAI's `.update()`/CDF-building code are potential graph-break sources (my inference; only compile training forward/loss, not compress/decompress).
- HF compile benchmarks table (T4, reduce-overhead): ResNet batch 4 13.871 vs 15.215 uncompiled (about 9%), units not stated. — [HF dataset CSV](https://huggingface.co/datasets/docs-benchmarks/compile-benchmarks/blob/main/reduce-overhead.csv)
- NVIDIA troubleshooting: with `reduce-overhead`, graph breaks fragment CUDA graphs and erase benefits. — [NVIDIA CUDA graph docs](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html)
- A 2026 arXiv found no compile benefit for tiny meta-interpreter graphs (non-CNN; low transfer). — [arXiv 2606.09930](https://arxiv.org/pdf/2606.09930)

### Inferences
- Where compile should help for this model: fusing GDN (x**2 + conv + rsqrt + mul) and the GMM likelihood elementwise chain, which are memory-bound; plausible total gain 5-20% (guess, unmeasured). Inductor/Triton supports sm75 (T4) but I did not find T4-specific compile results; Triton on Turing lacks some features (e.g. no bf16, fp16 ok).
- Fused Adam (`torch.optim.Adam(fused=True)`) on this small model (few tens of parameter tensors, ~5-6M params) saves only a few ms of launch overhead per step; found no T4 measurement. Free and numerically equivalent up to fp rounding.
- channels_last: only beneficial if convs run in fp16 tensor cores; in fp32 on T4 likely neutral or negative. GDN/rsqrt elementwise ops are layout-agnostic. EntropyBottleneck permutes channels internally, which could add copies with channels_last.
- Compile also changes RNG stream for uniform noise, so bitwise A/B parity is impossible; use statistical parity.

### Gaps
- No measured T4 numbers for torch.compile, CUDA graphs, fused Adam, or channels_last on a conv autoencoder with GDN. No GDN compile bug reports found (absence is not proof).

## Q5. Larger batch with scaled LR / gradient accumulation

### Takeaway
It is not equivalent: it changes the optimisation trajectory (fewer, less noisy updates, different Adam statistics). It also generally does not reduce wall-clock per sample on a compute-bound T4 and cannot meet "same iteration count" while saving time. I found no LIC study of batch-size scaling.

### Cited Findings
- Goyal et al.: linear LR scaling plus warmup preserves accuracy for ResNet up to batch 8192 on ImageNet, but they note large minibatches cause optimisation difficulties; this was SGD-momentum on classification, not Adam on RD loss. — [arXiv 1706.02677](https://arxiv.org/abs/1706.02677)
- No direct experiment varying batch size with matched LR in learned image compression found. A recognition-aware compression paper reports sensitivity to initial LR and instability at high lambda (lowering LR to 1e-5 helps). — [arXiv 2202.00198](https://arxiv.org/pdf/2202.00198)

### Inferences
- Gradient accumulation (k micro-batches) with the same LR is equivalent in expectation to a larger batch (fewer optimizer steps per sample), so 150k iterations of batch 8 != 150k/k steps of batch 8k. Keeping 150k optimizer steps at batch 8 and merely accumulating gives a different (larger-batch) experiment and costs k times the compute - no speed gain. For Adam the sqrt-LR scaling heuristic is common but unproven here. GPU utilisation at batch 8 x 256x256 already looks high (inferred), so larger batches buy little throughput per sample.
- Conclusion: do not use for a parity-preserving speedup; if used, treat as a different hyperparameter configuration needing its own validation.

### Gaps
- No LIC-specific evidence; no data on Adam + RD-loss large-batch behaviour.

## Q6. Cheap parity verification protocol

### Takeaway
No published LIC protocol found; the following is my recommended design (inference). Compare fp32 baseline vs optimised pipeline by statistical equivalence over several seeds at a short horizon, using held-out RD metrics, plus numerical-stability checks.

### Cited Findings
- Minnen&Singh and CompressAI-style training use batch 8, 256 crops, Adam 1e-4 (as in your setup); LR often dropped late in training in other works. — [arXiv 2007.08739](https://arxiv.org/pdf/2007.08739)
- CompressAI issue #92 shows NaN-bpp events happen even in fp32 and were mitigated with gradient clipping; so log NaN/skip counts for baseline as well. — [CompressAI #92](https://github.com/InterDigitalInc/CompressAI/issues/92)

### Inferences (protocol)
1. Step 0, numerics-only unit tests (minutes): same weights and same batch, compare fp32 vs AMP for (a) forward bpp, MSE, loss; (b) gradient cosine similarity per layer (expect >0.99) and relative norm ratio; (c) likelihood min value and count of zeros/inf; (d) GradScaler final scale and skipped steps over 500 steps. Fail if bpp relative error >~0.1-0.5% or any non-finite.
2. Step 1, speed benchmark (minutes): CUDA-event timed it/s after 50 warmup iterations, 200 measured, with data loading excluded (pre-load crops) so speed gains are not masked by the dataloader; also `torch.profiler` breakdown. Check peak memory.
3. Step 2, short A/B (e.g. 5-10k iterations at fixed lambda, same data order, 3 seeds per arm, since seed-to-seed noise is the yardstick): compare training loss curves (smoothed), and every 1-2k iterations evaluate on a fixed validation set (e.g. Kodak or 24-100 held-out crops, eval in fp32 with real `round` quantization, bpp from actual or likelihood-estimated rate): bpp, PSNR, and RD loss. Accept if the arm-difference in final RD loss/PSNR-at-equal-bpp is within the seed spread (rule of thumb: within ~0.02-0.05 dB PSNR and 0.2-0.5% bpp; these tolerances are my suggestion, calibrate from baseline seed variance). A BD-rate estimate needs several lambdas; at least one run at a low and one at a high lambda is advised because high-lambda/low-rate regime is where entropy-model precision matters most.
4. Step 3, parity of the late-training regime is not guaranteed by early runs: run one full or half-length (75k) run of the optimised pipeline at one lambda and compare to the existing fp32 curve, before committing all lambdas. Evaluate final model with the real entropy coder (actual bitstream size vs estimated) as an end-to-end check that fp16 training did not break the estimated-vs-actual bpp match.
5. Monitor: aux loss trajectory, fraction of skipped GradScaler steps, max activation inside GDN (overflow headroom vs 65504), min scale/likelihood.

### Gaps
- No published tolerance conventions for LIC A/B tests; the thresholds above are not sourced.

## Summary of confidence
- High (from source/docs): GDN is already conv1x1+rsqrt with fp32 params and no dtype handling; likelihood path is erfc-based with 0.11 scale bound and 1e-9 floor (below fp16 range); autocast list behaviours; EntropyBottleneck noise fixed for compile in 2026.
- Medium/low (inference): which parts are safe, speedup magnitude, compile gains, parity thresholds.
- Not found: any LIC AMP RD results; any T4 compile/fused-Adam/CUDA-graph measurements for this model class; LIC batch-size scaling study; GDN+compile bug reports.
