# Hardware and scheduling options to shorten a 150-250 T4-hour project (as of Nov 2026)

Scope note: Kaggle's official docs could not be fetched as text (JS-rendered), so the Kaggle facts below come from secondary sources and are flagged. All rental prices come from provider pages or third-party aggregators and move often. Speedup numbers are my own estimates (no training benchmark for this exact workload was found); they are marked as Inferences.

## Kaggle 2026 facts (quota, T4 x2 charging, sessions, P100, TPU)

### Takeaway
Kaggle does not publish a weekly GPU number in its docs; the figure is shown in account settings and is commonly reported as about 30 h/week (floating, "sometimes higher"). I found no authoritative statement on whether a T4 x2 session is charged once or twice. The evidence points to wall-clock session time, but this is unverified and must be checked in your own account.

### Cited Findings
- Kaggle does not publish the weekly GPU quota; it appears in account settings, floats, and users commonly report about 30 h/week. Sessions allow 12 h for CPU/GPU notebooks and 9 h for TPU — [gpuperhour.com, retrieved 21 Sep 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- Another aggregator lists 30 GPU h/week "sometimes higher depending on demand", a 60-minute idle timeout for interactive sessions, possible queueing at busy times, and a session length of 9 h (conflicts with the 12 h figure above) — [getaiperks.com](https://www.getaiperks.com/en/ai/free-gpu-students) (vendor that sells perks, so lower reliability). Disagreement: 9 h vs 12 h per session; treat 12 h as the better-supported figure (also in [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) and the Hackernoon author's "up to about 12 hours").
- Quota resets weekly (a 2023 university guide says Friday 8pm EDT/7pm EST) and an active interactive session counts toward the 30 h even if no process runs — [Notre Dame course page (2023)](https://www3.nd.edu/~dchiang/teaching/nlp/2023/kaggle.html), as relayed by search summary; older source.
- Kaggle offers "one P100 or two T4s"; most users use T4 x2 as one GPU. One team ran two independent jobs, one per GPU via CUDA_VISIBLE_DEVICES with a shared queue, finishing 20 runs in 5 h 9 min on two T4s; they note T4s are much faster in fp16 — [Hackernoon](https://hackernoon.com/a-journal-asked-for-three-seeds-we-had-a-laptop-and-euro0)
- Sources describe T4 x2 as 32 GB total VRAM (2 x 16 GB), one older source calls it "beta" — search summaries of [aimultiple](https://aimultiple.com/free-cloud-gpu) and [Kaggle/NVIDIA blog](https://developer.nvidia.com/blog/how-kaggle-makes-gpus-accessible-to-5-million-data-scientists/). Not confirmed on an official page.
- P100: no source says Kaggle retired it; Oct 2026 and Sep 2026 third-party guides still list P100 (and T4) as offered, with the menu changing over time. Google Cloud's P100 end-of-support (15 Sep 2026) applies to Compute Engine, not Kaggle — [Google Cloud P100 EOS](https://docs.cloud.google.com/compute/docs/eol/p100-eos)
- Linked Colab Pro / Pro+ subscribers reportedly got +15 / +30 Kaggle GPU h/week; "Kaggle may change or end this" (limited-time promo) — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), [getaiperks](https://www.getaiperks.com/en/ai/free-gpu-students)
- A claim that running two GPU notebooks at once drains quota twice as fast comes from a third-party blog (relayed by search summary); it concerns two notebooks, not one T4 x2 session.
- Phone verification is generally required to enable GPU/TPU/internet in notebooks (from model background knowledge in a fetch summary; no source found that quota size differs by verification level).

### Inferences
- Since quota displays as hours of session time and no source reports a 2x multiplier, a T4 x2 session is probably charged as 1 session-hour (i.e. you get 2 GPUs for the price of one). If true, T4 x2 gives ~2x the T4-hours per quota-hour (about 60 T4-h/week from a 30 h quota). Verify by watching the quota counter for 10-15 min in a T4 x2 session.
- If charged once, running two independent jobs (one per GPU) is the highest-yield trick: your base codec, fine-tunes, ablation chains and evaluation are separable jobs.
- Budget: 200 T4-h / 30 h/week = ~6.7 weeks on one-GPU-equivalent use; ~3.5 weeks if T4 x2 is used fully and charged once.

### Gaps
- Official Kaggle numbers (exact quota, concurrency limit for GPU sessions, RAM/CPU/disk, TPU weekly quota, phone-verification effect) not retrievable. Commonly quoted values I could not verify here: ~2 concurrent GPU sessions? (unverified, do not rely). Kaggle TPU quota is reported as floating (about 20 h/week historically); TPU is not useful for a PyTorch GDN/conv codec without a JAX/XLA port.
- Whether T4 x2 is charged once or twice: no direct source.

## Other free or low-cost GPU sources (2026)

### Takeaway
No free source matches Kaggle for GPU-hours. Useful additions are Colab pay-as-you-go (cheap T4 at about $0.12/h), Lightning's one-time credits (small), and TPU Research Cloud (free but requires a TPU port). Several formerly free options are closed or reduced.

### Cited Findings
- Colab (consumer): Pro is $9.99 for 100 units (about $0.10/unit as of Sep 2026). Measured burn rates (March 2026): T4 1.19 units/h (about $0.12/h), L4 1.71 (about $0.17/h), A100 40GB 5.40 (about $0.54/h); rates fluctuate (an Oct 2024 T4 measurement was 1.58) — [McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/), via search summary. Free tier: very limited, unpublished GPU limits, usually T4, max 12 h sessions — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- Colab Enterprise (Google Cloud, separate product, list price): T4 $0.42/h, L4 $0.672/h, A100 40GB $3.52/h, A100 80GB $4.71/h — [Google Cloud](https://cloud.google.com/colab/pricing)
- Colab Pro for Education free-year for US students (launched 18 Jul 2025) no longer accepts new signups — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits); getaiperks still advertises a free Pro plan for US students (conflict; the newer-dated gpuperhour note says closed).
- Lightning AI: sources disagree. Earlier 2026 sources: 15 free credits/month (about 22 T4-h on-demand or ~80 h interruptible), one free 4-CPU Studio, 4-hour restart limit. Newest source (21 Sep 2026): monthly credits replaced by a one-time grant of up to 30 credits (5 on signup, 25 with a card, expire after 12 months); T4 raised from $0.19 to $0.55/h, L4 $0.79, A100 40GB $2.19, H100 $4.50 — [UsagePricing](https://www.usagepricing.com/blueprint/activity/lightning-ai-2026-09-21-gpu-rate-card-reprice). Treat as poor value now.
- SageMaker Studio Lab: closed to new users; existing users get 4 h GPU sessions, max 4 h per 24 h — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- Paperspace Gradient free: M4000 (8 GB Maxwell), 6 h auto-shutdown, 5 GB storage — useless for this project — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- Hugging Face ZeroGPU: 5 min/day GPU on free accounts, PRO $9/mo for 40 min, Gradio Spaces only, 60 s default function — unsuitable for training — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- Azure for Students: $100 credit, no card needed (GPU eligibility not confirmed); GitHub Student Pack: Azure $100 plus Camber 5 GPU hours; DigitalOcean credit absent as of Sep 2026; Modal academic grants up to $10k for grad students/labs/researchers by application — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- getaiperks lists large headline figures (Google Cloud $350k, AWS $300k, Modal $50k, RunPod $1k) — marketing from a perk-selling vendor, mostly startup/research programs; not realistic for an individual student — [getaiperks](https://www.getaiperks.com/en/ai/free-gpu-students)
- Google TPU Research Cloud: free TPUs on rolling invitation after signup, but only the TPUs are free (VMs/storage billed), and participants agree to share research — [TRC](https://sites.research.google/trc/about/). The separate 2026 Google TPU awards call closed 30 Jun 2026 — [call page](https://prayat-pu.github.io/mike-lab/news/call-for-proposals-2026-google-awards-for-ml-research-and-education-with-tpus/)
- NVIDIA Academic Grant Program: officially not accepting new applications at the time of search; faculty-only eligibility — [NVIDIA](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program.md), [Vanderbilt summary](https://www.vanderbilt.edu/cttc/nvidia-academic-grant-program-ai-training-and-model-development/)

### Inferences
- Colab pay-as-you-go at about $0.12/T4-h means 200 T4-h is about $24 for a second, fully independent T4 stream (no speedup, but extra parallelism). Interactive/idle-timeout behavior makes 12 h unattended runs riskier than Kaggle's "save & run all".
- TRC would require rewriting to JAX/TF or PyTorch/XLA; likely not worth it for a GDN codec plus diffusion repo.
- A university HPC account or a supervisor's faculty-sponsored grant is the only route to a large free reduction; ask the department (not verifiable here).

### Gaps
- Colab consumer official pricing page not retrieved; unit rates from a third-party measurement.
- Free GPU eligibility of Azure Students credit unconfirmed. No 2026 source found for AWS Educate/GCP student GPU credits that an individual student can actually use for GPUs.

## Rental economics: price per hour and cost of the project

### Takeaway
Renting a RTX 4090 or A100 is cheap: the 50-80 T4-h diffusion run costs roughly $6-20 and finishes in about 10-20 h of wall-clock, versus 50-80 h of T4 quota. The entire 200 T4-h project is roughly $20-60 on a 4090/A100, at my assumed speedups (unverified).

### Cited Findings
- RunPod pricing page (per-second billing), Secure / Community $/h: A100 PCIe 80GB 2.09 / 1.19; A100 SXM 80GB 2.19 / 1.39; L40S 1.09 / 0.79; RTX 4090 0.89 / 0.34; L4 0.69 / 0.44; A40 0.59 / 0.35; H100 PCIe 2.89 / 1.99; H100 SXM 4.39 / 2.69; RTX A6000 0.69 / 0.33. Community figures are "from" prices — [RunPod](https://www.runpod.io/gpu-cloud/pricing). Other RunPod pages and third parties show higher numbers (H100 $3.99, 4090 $0.89, L40S $0.99-1.09, A100 80GB $1.79) — conflicting by tier/date.
- Vast.ai (16 Nov 2026 snapshot): median RTX 4090 $0.50/h (cheapest $0.27), median A100 SXM4 $0.84 (cheapest $0.35), median H100 SXM $1.93 — [Markaicode](https://markaicode.com/pricing/vast-ai-pricing/) (methodology undocumented). A July 2026 API snapshot shows 4090 $0.136-0.349 — via [Vast pricing search summary](https://www.activepieces.com/blog/vast-ai-gpu-pricing-per-hour-in-2026-and-extra-fees.md). Marketplace hosts vary in reliability and bandwidth.
- Thunder Compute: A100 80GB $0.78/h (another page $1.09), H100 $1.38/h, L40 $0.79 — [Thunder pricing](https://discord.thundercompute.com/pricing), [Spheron](https://www.spheron.network/blog/thunder-compute-pricing-2026/); conflicting values.
- Lambda A100 80GB about $1.99/h (Aug 2026, one tracker; others differ); Modal A100 80GB about $2.50/h, H100 about $3.95/h — [computeprices](https://computeprices.com/compare/lambda-vs-thundercompute), [Hivenet](https://www.hivenet.com/post/nvidia-a100-pricing-guide), [tech-insider](https://tech-insider.org/gpu-cloud-providers-serverless-gpu-pricing-modal-runpod-nebius-2026/)
- Lightning AI (21 Sep 2026): T4 $0.55, L4 $0.79, A100 40GB $2.19, H100 $4.50 per GPU-h — [UsagePricing](https://www.usagepricing.com/blueprint/activity/lightning-ai-2026-09-21-gpu-rate-card-reprice)
- No published benchmark for training a fp16 conv U-Net or an fp32 GDN codec on T4 vs A100/4090 was found; the searched benchmarks are inference only. A vendor guide claims A100 costs 2x a 4090 for 4x throughput (no methodology). RunPod states its benchmark page is inference, not training guidance — [RunPod benchmarks](https://www.runpod.io/gpu-benchmarks)
- A10 and the Colab/Lightning hourly rates: no reliable 2026 A10 price found.

### Inferences (my estimates, not sourced; a 10-minute benchmark on each rental is strongly recommended)
Assumed speedup vs T4 (T4 has 320 GB/s bandwidth, ~65 fp16 peak but typically 20-30 realized; 8 fp32 TFLOPS):
- (a) fp16 conv U-Net, batch 32 at 128px: A100 about 4-6x; RTX 4090 about 3.5-5x (24 GB fits); L40S about 3.5-5x; H100 about 7-10x; L4 only about 1-1.5x (similar memory bandwidth to T4).
- (b) fp32 codec with GDN: convs benefit from TF32 (on by default for cuDNN convs in PyTorch; matmul TF32 must be enabled) but GDN and elementwise layers are memory-bound, so the speedup is closer to the bandwidth ratio: A100 about 3-5x, 4090 about 3-5x (82 fp32 TFLOPS but 1 TB/s), H100 about 5-8x. Switching to TF32/bf16 changes numerics, so this is not "same results" strictly.

Cost of the 65 T4-h (range 50-80) diffusion job (midpoint; community "from" prices are optimistic, use secure/median as upper bound):

| GPU | Assumed speedup | Hours | Price $/h (low-high) | Cost |
|---|---|---|---|---|
| T4 on Kaggle | 1x | 65 (about 2.2 weeks of quota, or about 1.1 weeks if T4 x2 charged once and two jobs) | 0 | $0 |
| T4 on Colab PAYG | 1x | 65 | 0.12 | about $8 |
| L4 (RunPod) | 1.25x | 52 | 0.44-0.69 | $23-36 |
| RTX 4090 | 4x | 16 | 0.34-0.89 | $6-14 |
| L40S | 4x | 16 | 0.79-1.09 | $13-18 |
| A100 80GB | 5x | 13 | 0.78-2.09 | $10-27 |
| H100 PCIe | 8x | 8 | 1.38-2.89 | $11-23 |

Whole project (about 200 T4-h): 4090 about 50 h, about $17-45; A100 about 40 h, about $31-84; H100 about 25 h, about $35-72. Those are sequential totals; with the dependency graph (about 20 base + 6 first fine-tune + 65 diffusion on the critical path = about 90 T4-h) the critical path on a 4x GPU is about 22 h, so calendar time is bounded by that, with the other fine-tunes, ablations and evaluation running in parallel on Kaggle.

### Gaps
- No verified training benchmarks. Real speedups can be 30-50% lower if the data loader, CPU or checkpoint I/O is the bottleneck (the rental's vCPU count matters).
- Vast.ai/marketplace prices are volatile and tier-dependent; A10 price missing.

## Practical patterns (spot, resumable, cost control) and Kaggle T4 x2 usage

### Takeaway
Because state lives on Hugging Face and runs resume from checkpoints, spot/interruptible rentals and free sessions are both safe; the main cost control is per-second billing, auto-stop, and checkpointing frequently. On Kaggle, running two independent jobs, one per GPU, is the lowest-risk way to use T4 x2; DDP gives less than 2x and ties both GPUs to one job.

### Cited Findings
- Kaggle: one worker per GPU using CUDA_VISIBLE_DEVICES, no DataParallel, queue-based and resumable by re-running the cell; sessions can die — [Hackernoon](https://hackernoon.com/a-journal-asked-for-three-seeds-we-had-a-laptop-and-euro0)
- A Hugging Face forum report that notebook_launcher with num_processes=2 on Kaggle launched on one GPU (DDP launch pitfalls in notebooks) — [HF forum](https://discuss.huggingface.co/t/notebook-launcher-set-num-processes-2-but-it-say-launching-training-on-one-gpu-in-kaggle/27430)
- RunPod bills per second with no commitment; Thunder Compute per minute/second — see pricing citations above.
- Lightning spot (interruptible) is no longer cheaper than on-demand for T4 — [UsagePricing](https://www.usagepricing.com/blueprint/activity/lightning-ai-2026-09-21-gpu-rate-card-reprice)

### Inferences
- Suggested split across free sessions: Kaggle GPU 0 = critical path (base codec then first fine-tune then diffusion); GPU 1 = other four fine-tunes and ablation chains; evaluation as dependencies finish. Different jobs have different VRAM needs; 16 GB per T4 is the limit.
- DDP on T4 x2 (PCIe, no NVLink) typically scales about 1.6-1.9x with enough batch per GPU, but changes the effective batch size and can alter results; independent jobs avoid this and preserve results exactly.
- Spot/interruptible rental: save a checkpoint to HF every 15-30 min; use an idempotent launcher script (pull latest checkpoint then train); set a spend cap and auto-terminate. Bring up a rented 4090/A100 only for the 65 T4-h diffusion run (critical path) and keep everything else on free Kaggle.
- For legitimate quota expansion: combine separate services (Kaggle + Colab PAYG + a few dollars of rental), not extra accounts on one service. Do not make multiple accounts at one provider; that violates terms.

### Gaps
- Kaggle concurrent GPU-session limit and whether T4 x2 sessions count once: not confirmed from official sources.
- Whether DDP on Kaggle T4 x2 gives specific speedups for this repo: no data.
