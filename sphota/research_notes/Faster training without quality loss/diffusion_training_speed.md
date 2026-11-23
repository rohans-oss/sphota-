# Faster training of a conditional diffusion U-Net on Kaggle T4 x2 without quality loss (as of 2026-11)

Scope note: web search returned almost no T4-specific *training* benchmarks. Items below are either cited, or explicitly labelled "uncited general knowledge / needs measurement". No invented numbers.

## Q1. Real measured gains on T4/Turing of compile, CUDA graphs, channels_last+AMP, fused AdamW/EMA, SDPA, cudnn.benchmark etc.

### Takeaway
No reliable published T4 training benchmark for a diffusers UNet2DModel was found. Existing evidence says torch.compile gives small gains on T4 (single-digit % in diffusers inference tables; ~none in some forum reports), channels_last+fp16 helps tensor-core GPUs in general (~8-35% on V100-class), and fused AdamW is documented as faster than foreach but with no cited number. Everything must be measured with a 200-step micro-benchmark on the actual T4.

### Cited Findings
- Diffusers' own benchmark tables include T4 rows; SDPA is about as fast as xFormers, and torch.compile adds only a few percent on T4 (roughly 3-8% at batch 4-16, fp16 *inference*, older PyTorch/diffusers versions). — [Diffusers torch2.0 docs](https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0) (figures as summarised by search; verify against the page)
- PyTorch blog on accelerated diffusion: up to 49% inference speedup vs. original without xFormers, 39% vs. with xFormers, depending on GPU and batch size (excluding compile time); table includes T4 but these are pipeline inference results, not training. — [PyTorch blog](https://pytorch.org/blog/accelerated-generative-diffusion-models/)
- Forum report of almost no torch.compile speedup on a T4; a commenter attributes it to T4 being compute-limited rather than memory-bandwidth-limited (opinion, not benchmark). — [PyTorch forums](https://discuss.pytorch.org/t/no-speedup-from-torch-complile/175497)
- HF transformers docs: on T4, compile gains for vision models are small to none (e.g. ViT bs4 43.653 -> 43.626 ms; Segformer 45.327 -> 42.445 ms), inference. — [HF perf_torch_compile](https://huggingface.co/docs/transformers/perf_torch_compile)
- Compile happens on the first training step (slow first step); changing input shapes triggers recompiles. An SDXL UNet user reported repeated recompilation with varying resolutions (anecdote; the 2.3x claim there is unverified and not T4). — [HF torch_compile docs](https://huggingface.co/docs/transformers/v5.12.0/torch_compile); [DEV post](https://dev.to/elise_moreau/torchcompile-recompiled-our-sdxl-unet-38-times-in-production-o91)
- torch.compile modes: `reduce-overhead` uses CUDA graphs (helps small batches / Python-overhead-bound; extra memory; only for graphs that don't mutate inputs); `max-autotune` benchmarks kernels (longer compile) and in current docs enables CUDA graphs by default; `max-autotune-no-cudagraphs` is the fallback. — [torch.compile docs](https://docs.pytorch.org/docs/main/generated/torch.compile.html)
- channels_last: biggest gains on NVIDIA tensor-core GPUs in reduced precision; PyTorch's AMP ResNet-50 test reported >22%; fast.ai reports 8-35% on V100 and says it works on Volta/Turing+. Mixed layouts ("channel thrashing") from ops without channels_last support can erase gains; cuDNN may still choose non-tensor-core algorithms. Some users report slowdowns. — [PyTorch memory format tutorial](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html); [fast.ai channels_last](https://docs.fast.ai/callback.channelslast.html.md); [forum slowdown](https://discuss.pytorch.org/t/channels-last-format-convolution-is-slower-than-normal-nchw/107586)
- AdamW docs: foreach and fused are typically faster than the for-loop; fused takes priority and cannot be combined with foreach=True; default is foreach because fused is newer; foreach uses ~sizeof(params) extra peak memory. For a ~35M-param model the optimizer step is a small share of step time anyway. — [torch.optim.AdamW docs](https://docs.pytorch.org/docs/2.2/generated/torch.optim.AdamW.html)
- Databricks/MosaicML SD2 cost write-up: xFormers/flash attention, precomputed latents, low-precision LayerNorm/GroupNorm (norms run fp32 by default), FSDP, and scheduled EMA cut the projected cost from <$50k (7.45 days) to <$42k (6.79 days, 21,000 A100-hours); scheduled EMA avoids EMA overhead for 96.5% of training. Different hardware (A100) and scale. — [Databricks blog](https://www.databricks.com/blog/diffusion); [Databricks SD2 post](https://www.databricks.com/blog/stable-diffusion-2)

### Inferences
- With only a mid-resolution (1/4 res = 32x32 = 1024 tokens) attention at 128px, batch 16-32, attention is likely a minority of step time; convs/GroupNorm dominate. So compile/SDPA tweaks likely give modest gains; expect the largest cheap wins from (a) channels_last + fp16 if GroupNorm does not thrash layouts, (b) cheaper EMA / optimizer (foreach/fused), (c) running the frozen codec in fp16/channels_last, (d) DDP on 2 GPUs. Rank by measurement, not by this note.
- Uncited general knowledge (verify): mem-efficient SDPA backend supports Turing (sm75) in fp16; FlashAttention backend in PyTorch SDPA requires sm80+, so on T4 SDPA falls back to mem-efficient or math. xFormers on T4 is unlikely to beat SDPA mem-efficient (supported by diffusers table above showing parity). Force with `torch.nn.attention.sdpa_kernel` to check which backend is used. `torch.backends.cudnn.benchmark=True` is safe with fixed 128x128 shapes; `zero_grad(set_to_none=True)` is default in torch 2.x; GradScaler overhead is small (unscale + inf check; fused AdamW with scaler supported). None have T4 numbers.
- torch.compile numerically changes kernel fusion (tiny fp differences) but not the math; it is "identical up to floating-point noise".

### Gaps
- No T4 training benchmark for diffusers UNet2DModel with compile/channels_last/fused AdamW; no graph-break report for UNet2DModel found; no compile-time figures on T4; no "known T4 regressions" source found; no source on GroupNorm+channels_last behaviour in current PyTorch; no cited source confirming SDPA backend support matrix (docs page fetch failed with a redirect); no cited xFormers-on-T4 training number.

## Q2. Using both GPUs (DDP on T4 x2 over PCIe)

### Takeaway
Same-global-batch DDP is mathematically equivalent to single-GPU batch 32 (with the caveats below), but the only T4 x2 Kaggle data point found (a tiny 4M-param transformer) showed DDP *slower* than single GPU. A ~35M-param U-Net has far more compute per step, so scaling should be much better, but there is no direct measurement; it must be measured.

### Cited Findings
- Two T4s on Kaggle over PCIe (no NVLink), 4M-param nanoGPT: single GPU 22.03 ms/step; DDP 26.36 ms/step (compute 15.12 ms, communication 11.24 ms), 42.6% communication overhead, 41.8% scaling efficiency. — [DEV Community: DDP is not always faster](https://dev.to/thokozani_buthelezi_2cd41/ddp-is-not-always-faster-23mh)
- PCIe-only 4x RTX A6000: plain DDP gave ~2x on four cards; bf16 gradient compression plus accumulation ~3.6x (single anecdotal run; bf16 not available on T4). — [PyTorch forums](https://discuss.pytorch.org/t/ddp-on-a-pcie-only-4x-rtx-a6000-no-nvlink-plain-ddp-gave-2x-bf16-gradient-compression-plus-accumulation-gave-3-6x/225650)
- HF docs: disabling NVLink (NCCL_P2P_DISABLE=1) slowed 2-GPU DDP from 101 s to 131 s on a larger model; slow interconnect can make DDP slower than one GPU. — [HF performance docs](https://huggingface.co/transformers/v4.8.0/performance.html)
- DDP buckets gradients and overlaps allreduce with backward; little backward compute means little to hide behind. Tuning knobs mentioned: bucket_cap_mb, bigger batches, mixed precision. The blog cited for tuning is on 8xH100 NVLink and does not transfer. — [karthmnz DDP blog (H100 only; no T4 data)](https://karthmnz.github.io/blog/2025/DDP.html)
- Kaggle notebook pitfalls: CUDA must not be initialised before launching worker processes; put all code in one cell / a function passed to Accelerate `notebook_launcher`; old Accelerate versions launched on one GPU in Kaggle; NCCL_P2P_DISABLE mentioned in a non-authoritative dataset card. — [fast.ai forum](https://forums.fast.ai/t/how-to-use-fastai-with-2-t4-gpus-on-kaggle/112173); [HF forum](https://discuss.huggingface.co/t/notebook-launcher-set-num-processes-2-but-it-say-launching-training-on-one-gpu-in-kaggle/27430); [PyTorch forum](https://discuss.pytorch.org/t/help-with-ddp-in-kaggle-notebook/213369); [Accelerate notebook docs](https://huggingface.co/docs/accelerate/v1.0.0rc1/basic_tutorials/notebook)

### Inferences
- Gradient volume: ~35M params in fp32 = ~140 MB per allreduce (uncited arithmetic). At typical PCIe 3.0 x16 effective ~10 GB/s ring bandwidth through host (unverified for Kaggle), that is on the order of tens of ms per step, comparable to or possibly a large fraction of a T4 step of batch 16 at 128px. So efficiency could land well below 90%; the real figure is unknown. Mitigations to test: `gradient_as_bucket_view=True`, bucket_cap_mb 25-100, fp16 gradient compression hook (`fp16_compress_hook`, changes allreduce precision slightly: not strictly identical), `no_sync` + accumulation does not help here since global batch is fixed.
- Equivalence caveats: (1) loss must be averaged over global batch (DDP averages grads, so 2x16 mean-loss equals 1x32 mean-loss); (2) GroupNorm is per-sample so no BatchNorm sync problem; (3) per-rank RNG seeding for noise/timesteps/dropout/crops must differ per rank; (4) GradScaler: use one per rank, inf-check is per rank but DDP-synced grads make skips consistent in practice; clipping after allreduce; (5) EMA updated identically on each rank or on rank 0 only. Random streams will differ from single-GPU run, so runs are statistically but not bitwise identical.
- The frozen codec runs per rank on its 16 samples, so it is parallelised too (benefit).
- Compare against the alternative: two independent single-GPU jobs (e.g. two ablations or seeds) uses both GPUs with perfect efficiency; DDP only pays off for shortening the single longest job.

### Gaps
- No measured DDP scaling for a conv U-Net on T4 x2; no Kaggle-specific NCCL tuning guidance from an authoritative source; P2P availability between the two Kaggle T4s not verified.

## Q3. Same-quality-in-fewer-steps methods (same network)

### Takeaway
Min-SNR-gamma (gamma=5) has the strongest evidence, applies directly to epsilon prediction, and is a one-line loss reweighting; but it changes the objective weighting, so it is *not* identical training and must be validated. Timestep sampling methods (SpeeD, adaptive non-uniform sampling) claim large speedups but are less validated for small conditional pixel-space U-Nets. v-prediction / zero-terminal-SNR / EDM-style changes alter the parameterisation and are not drop-in.

### Cited Findings
- Min-SNR: epsilon-prediction weight w_t = min(SNR(t), gamma)/SNR(t) = min(gamma/SNR(t), 1); default gamma=5. Treats timesteps as tasks with conflicting gradients. Headline 3.4x faster to FID 10 (ViT-B, ImageNet 256 latent). — [arXiv 2303.09556](https://arxiv.org/html/2303.09556v3)
- Min-SNR on a ADM-style UNet (ImageNet 256 latent, base ch 192; much bigger than yours), epsilon prediction FID: 8.55 -> 7.32 at 200K iterations; 4.21 -> 4.14 at 1M iterations. So for eps-prediction the gain on a UNet is modest and shrinks with training length. gamma sweep for UNet eps: 1/5/10/20 gives 4.30/4.14/4.14/4.12. — [arXiv 2303.09556](https://arxiv.org/html/2303.09556v3)
- Min-SNR paper has no CIFAR results; pixel-space ablation is ViT-B on ImageNet 64; applied to EDM only with limited compute. — [arXiv 2303.09556](https://arxiv.org/html/2303.09556v3)
- Adaptive non-uniform timestep sampling (Kim et al., CVPR 2025): gradient variance differs across timesteps; samples timesteps adaptively; reports significantly faster convergence. — [arXiv 2411.09998](https://arxiv.org/html/2411.09998v2)
- SpeeD (Wang et al., 2024): asymmetric timestep sampling plus change-aware weighting; claims 3x acceleration across architectures/datasets. — [arXiv 2405.17403](https://www.arxiv.org/abs/2405.17403)
- Log-SNR importance sampling is equivalent to a modified noise schedule; helpful when sampling concentrates near logSNR=0. — [arXiv 2407.03297](https://arxiv.org/pdf/2407.03297v2.pdf)
- Zero terminal SNR + v-prediction (Lin et al., WACV 2024): fixes train/inference mismatch at final timestep (mainly brightness issues in Stable Diffusion); epsilon-prediction cannot be used at zero terminal SNR (division by zero), so v-prediction is needed. Authors found visual quality similar to epsilon after fine-tuning. This is a quality/bias fix, not a speed-up. — [arXiv 2305.08891](https://arxiv.org/abs/2305.08891)
- EDM2 (Karras et al., CVPR 2024): post-hoc EMA (reconstruct arbitrary EMA lengths from stored snapshots, no extra training runs), plus architecture changes (magnitude-preserving layers) that change the network, so not applicable under "same network". — [arXiv 2312.02696](https://arxiv.org/abs/2312.02696v1); [NVIDIA blog](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models)
- Stretching Each Dollar: micro-budget diffusion training from scratch (arXiv 2407.15811), relevant general reference; details not extracted here. — [arXiv 2407.15811](https://arxiv.org/pdf/2407.15811)

### Inferences
- Under "same model, data, steps" the baseline is already a cosine schedule, which has a milder SNR range than linear; epsilon-loss with cosine is already near Min-SNR-like balance at high-SNR ends? (Unverified; do not assume.) Min-SNR's measured gain for eps-prediction at long training (200K steps is your budget) was small in the cited UNet table, so expect a modest convergence benefit; the stronger benefit is for x0/v parameterisations.
- Classification for the report: Min-SNR weighting, timestep sampling, EDM preconditioning, v-pred, zero-terminal-SNR, power-function EMA all change the optimisation or objective, so none is "identical". Post-hoc EMA / EMA schedule changes only the EMA output, not the raw weights (safest of the group: can store snapshots and pick EMA length afterwards at no training cost). Progressive/multi-resolution training changes the data distribution per step and the 128px target; not recommended under the constraint (no source retrieved).
- Risk: with CFG dropout and conditional restoration (codec residual conditioning), min-SNR changes the effective weighting of low-noise timesteps, which matter for fine detail/LPIPS; verify with perceptual metrics, not only loss.

### Gaps
- No source found testing these methods on small conditional pixel-space restoration/codec-refinement U-Nets; no 2025-2026 independent replication of SpeeD or adaptive sampling checked; v-pred vs eps convergence-speed comparisons and progressive-resolution evidence not retrieved; power-function EMA formula not extracted (see EDM2 Sec. 3).

## Q4. Reusing computation: frozen codec conditioning

### Takeaway
No source found directly addressing this. Reasoning-only options are listed; all need measurement.

### Cited Findings
- MosaicML precomputed VAE/CLIP latents for SD2 (3,784 A100-hours one-off; computing online instead would cost ~1.4x more), a precedent for caching frozen-model outputs, though their data had no random-crop dependence. — [Databricks blog](https://www.databricks.com/blog/stable-diffusion-2); [mosaicml/diffusion](https://github.com/mosaicml/diffusion)

### Inferences (uncited, flag as reasoning)
- Run codec under `torch.inference_mode()` / `no_grad` with fp16 autocast and channels_last. fp16 changes the conditioning image slightly (not bit-identical; test PSNR between fp32 and fp16 codec outputs and the downstream effect, since the diffusion model learns to correct codec artefacts specifically).
- If the codec is fully convolutional with a stride-s encoder and the crop offsets are multiples of the total downsampling stride, cropping the latent y of the full image is nearly equivalent to encoding the crop, except at borders (padding/receptive-field effects) and for any entropy-model/quantisation context that depends on position. Flip commutes with a conv codec only if the filters are flip-equivariant (generally not exactly). So an exact equivalent requires verification; a cheaper approximation is caching reconstructions of full images per epoch/offset-grid.
- Cost share of codec is unknown; measure: time(codec fwd) vs time(UNet fwd+bwd). Codec forward in fp32 on 128x128x32 could be a non-trivial share on a T4 (fp32 only ~8 TFLOPS peak, vs fp16 tensor cores). Profile first.

### Gaps
- No measurements; cost share of the codec not available; no literature on caching with random crops found.

## Q5. Gradient checkpointing, larger batch + LR scaling, low-precision (torchao) on Turing

### Takeaway
Not researched with sources in this pass (search budget spent); treat as low priority. Checkpointing only trades compute for memory (about +30% compute, uncited general knowledge), useless if batch 32 at 128px fits in 16 GB. Larger batch changes the optimisation and is not identical.

### Cited Findings
- None retrieved.

### Inferences
- Uncited: torchao fp8/int8 training paths need sm89/sm90 or have no tensor-core benefit on Turing (T4 has fp16 and int8/int4 tensor cores but no bf16/fp8); not expected to help here.
- Keep batch 32 per spec; larger batch + LR scaling changes the optimisation per step and is outside "identical".

### Gaps
- No sources for torchao-on-Turing, checkpointing overhead, or LR-scaling for diffusion.

## Q6. Parity-verification protocol (proposed; derived from the above, not from a cited source)

### Takeaway
Separate "numerically equivalent" changes (compile, channels_last, fused optimizer, foreach EMA, DDP with same global batch, fp16 codec) from "objective-changing" changes (Min-SNR, timestep sampling, v-pred, schedule changes), and gate the latter on quality metrics.

### Inferences
- Step 0 (throughput): fixed seed, 50 warmup + 200 timed steps, `torch.cuda.synchronize()`, report median ms/step and peak memory; run each toggle singly and combined. Exclude compile time but report it separately (200k steps amortise it).
- Step 1 (bitwise-ish parity for numerically equivalent changes): same seed, same batch, same init; compare forward loss and gradient norm at step 0 (rel. diff target <1e-3 in fp16 autocast; unscaled differences of this size are expected from kernel changes), then 500-1000 step loss curves; compare smoothed loss (window 100) within seed-to-seed noise: first measure noise by running baseline with 3 seeds, then require the change to lie within that band.
- Step 2 (DDP): confirm effective global batch 32, per-rank different RNG, grad norm after allreduce equals single-GPU for identical data split (feed the same 32 samples split 16/16: gradients should match to ~1e-5 fp32 / fp16 noise).
- Step 3 (quality gate for non-identical changes): short A/B (e.g. 20-30k steps, EMA weights) with a fixed set of validation images, fixed sampler/seed/steps, compare PSNR/LPIPS vs target and FID or KID on enough samples (FID needs thousands of samples to be stable; with small sets use KID/LPIPS), plus paired comparison on the same noise seeds. Accept if metric is within baseline seed-to-seed variation (tolerances must be set from measured 2-3 baseline seeds; none can be cited).
- Step 4: also check EMA-model samples at several checkpoints; Min-SNR reports its advantage early in training and it shrinks later (see Q3 UNet table), so run the A/B long enough to see convergence rather than only early steps.

### Gaps
- No published tolerance thresholds for such comparisons were found; they must be empirically calibrated.
