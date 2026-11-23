# TrueSight: A Faithful Generative Image Codec

> *Send a photo at 1/20th of the data, and never let the AI change what matters.*

**Primary course:** Generative AI & Diffusion Models. The project also integrates Unsupervised Learning, Supervised Learning, Advanced ML and Neural Networks.

**Status:** project skeleton only. The folder structure and stub modules are in place and no model code is written yet. Implementation follows the 12-week plan below.

---

## 1. Problem

Images and video make up most mobile traffic, and data and storage are still expensive. Classical codecs (JPEG, WebP) fall apart below roughly 0.1 bpp, producing blocks and blur. Learned codecs do better but turn blurry at extreme compression. Generative codecs (GAN or diffusion decoders) look excellent at 0.01 to 0.05 bpp, but they **hallucinate**: they invent plausible detail that was never transmitted.

| What the user sent | What a hallucinating codec can deliver | Consequence |
|---|---|---|
| Medicine label "5 mg" | Sharp label reading "8 mg" | Health risk |
| Invoice total "4,750" | Clean digits "4,150" | Financial or legal dispute |
| Photo of a child | Realistic face of a slightly different child | Loss of trust |
| Dashcam licence plate | Plausible but wrong plate | Evidence is worthless |

### Formal problem statement

Given an image `x` and a bit budget `R`, produce a bitstream `b` with `|b| <= R` and a decoder `D` such that `x_hat = D(b)` is:

1. **Realistic:** low perceptual distance (LPIPS, FID) to natural images.
2. **Faithful:** protected regions (text, digits, faces) are preserved, measured by OCR character error rate and face-identity similarity.
3. **Consistent:** re-encoding `x_hat` gives the same quantised latent that was transmitted.
4. **Honest:** every region where the decoder had to invent detail is flagged in an uncertainty map, with an optional safe (non-generative) fallback.

## 2. Solution

TrueSight has three promises: **small**, **beautiful**, and **faithful and honest**.

| # | Block | What it does |
|---|---|---|
| 1 | VAE codec + GMM entropy model | Turns the image into a small integer latent and codes it with an arithmetic coder. Rate = -log2 p(latent). |
| 2 | Conditional diffusion decoder | Generates realistic texture conditioned on the coarse decode. Classifier-free guidance (CFG) trades realism against fidelity. |
| 3 | Faithfulness layer | Text/face ROI coding, **bin-consistency guidance**, ROI-lock guidance. |
| 4 | Honesty layer | Seed-variance and consistency maps, anomaly detection, risk classifier, safe fallback. |
| 5 | Intelligence layer | Rate control (regression, XGBoost), mode recommender (kNN, decision tree), failure mining (association rules). |

**Bin-consistency guidance** is the main contribution. During every denoising step the sample is nudged so that its re-encoded latent falls into the same quantisation bins as the transmitted latent. Any image whose latent rounds to the transmitted integers is, by definition, consistent with the bitstream.

## 3. Architecture

### Encoder (sender)

```
input image x (H x W x 3)
  |
  +--> [Protect]       text detector (EasyOCR/CRAFT) + face detector (MTCNN)
  |                    + risk model (logistic regression on patch features)
  |                    --> protected boxes (aligned to 16 px)
  |
  +--> [Rate control]  features --> XGBoost --> choose lambda index (mode)
  |
  +--> [VAE codec, low lambda]  y = g_a(x), z = h_a(y)
  |        z --> factorized entropy model --> bitstream Bz
  |        h_s(z_hat) --> GMM params (w, mu, sigma) per latent element
  |        round(y) --> arithmetic coder with GMM CDF --> bitstream By
  |
  +--> [ROI codec, high lambda]  each protected crop --> bitstream Broi_i
  |
  v
.tsi file = header + Bz + By + [box_i, Broi_i] + seed
```

### Decoder (receiver)

```
.tsi file
  --> decode z_hat, then GMM params, then y_hat (bit-exact)
  --> coarse image x_c = g_s(y_hat)                  ("honest" reconstruction)
  --> decode protected crops x_roi and mask M
  --> conditional diffusion (DDIM, 25-50 steps, CFG scale w)
        at each step: x0_hat --> bin-consistency guidance (match y_hat bins)
                             --> ROI-lock guidance (match x_roi inside M)
  --> paste protected crops (feathered mask)
  --> hallucination map: seed variance + consistency error + risk model
  --> output: realistic image x_hat + uncertainty map + safe fallback x_c
```

### `.tsi` bitstream format

| Field | Size | Meaning |
|---|---|---|
| magic | 4 B | ASCII `TSI1` |
| version | 1 B | Format version (1) |
| height, width | 2 + 2 B | Original image size |
| mode | 1 B | Index of the lambda / model used (0 = smallest) |
| seed | 4 B | Diffusion seed, so decoding is reproducible |
| guidance scale | 1 B | CFG scale x 10 (15 = 1.5) |
| len(Bz), len(By) | 4 + 4 B | Byte lengths of the two latent streams |
| Bz, By | variable | Entropy-coded hyper-latent and latent |
| n_boxes | 1 B | Number of protected regions (0-255) |
| per box: x, y, w, h, len | 2+2+2+2+4 B | Box (multiples of 16) and stream length |
| Broi_i | variable | High-quality codec stream for each box |

The header costs about 30 bytes plus 12 bytes per box. It is included in every reported bpp.

## 4. Repository layout

```
truesight/
  README.md
  requirements.txt
  configs/              codec.yaml, diffusion.yaml, eval.yaml
  data/
    raw/                downloaded datasets (never edited)
    processed/          crops, splits, file lists
  truesight/
    data.py             datasets and crops
    metrics.py          bpp, PSNR, MS-SSIM, LPIPS, CER, face similarity, BD-rate
    baselines/          jpeg.py, pca_codec.py, vq_codec.py
    codec/              layers.py, vae_codec.py, entropy_gmm.py, bitstream.py, train_codec.py
    diffusion/          schedule.py, unet.py, train_diffusion.py, sample.py, guidance.py
    protect/            detect.py, roi.py
    analysis/           features.py, risk_model.py, hallucination.py, recommender.py, rules.py, rate_control.py
    app/                cli.py, api.py, ui.py
  scripts/              prepare_data.py, eval_rd.py, eval_faithfulness.py, make_figures.py
  experiments/          one folder per run: config, logs.csv, checkpoints, figures
  tests/                test_metrics.py, test_bitstream.py, test_schedule.py
  report/               figures and the final report
```

| Component | File(s) | Week |
|---|---|---|
| Data pipeline | `truesight/data.py`, `scripts/prepare_data.py` | 1 |
| Metrics | `truesight/metrics.py` | 1 |
| Classical baselines (JPEG, WebP, PCA, VQ) | `truesight/baselines/*.py` | 2 |
| Feature extraction and size predictor | `truesight/analysis/features.py` | 2 |
| Autoencoder and VAE codec | `truesight/codec/vae_codec.py` | 3-4 |
| GMM entropy model and bitstream | `truesight/codec/entropy_gmm.py`, `bitstream.py` | 4 |
| Diffusion schedule, U-Net, sampler | `truesight/diffusion/*.py` | 5-6 |
| ROI protection | `truesight/protect/*.py` | 7 |
| Faithfulness guidance | `truesight/diffusion/guidance.py` | 8 |
| Hallucination detector | `truesight/analysis/hallucination.py` | 9 |
| Rate control, recommender, rules | `truesight/analysis/*.py` | 10 |
| CLI, API, UI | `truesight/app/*.py` | 12 |

## 5. How each course is used

| Course | Topic | Where in TrueSight |
|---|---|---|
| 5 GenAI | MLE | Entropy model trained by max likelihood: rate = -log2 p(y_hat) (`codec/entropy_gmm.py`) |
| 5 GenAI | GMM | Discretised Gaussian mixture (K=3) per latent element; ablation K=1 vs K=3 |
| 5 GenAI | Autoencoders / VAE | Deterministic AE baseline; codec is a VAE with loss = rate + lambda x distortion |
| 5 GenAI | Diffusion forward / backward | Own noise schedules, epsilon-prediction U-Net, DDPM training, DDIM sampling |
| 5 GenAI | Conditioning and guiding | Coarse-decode conditioning, CFG, bin-consistency and ROI-lock guidance |
| 4 NN | MLP, perceptron, CNN | Patch hallucination-risk baselines, conv encoder/decoder, U-Net |
| 4 NN | Activations, optimizers | GDN vs ReLU, Adam vs AdamW, cosine LR with warm-up, EMA |
| 1 Unsup. | Data, distances | Crop pipeline, PSNR/MS-SSIM/LPIPS/CER/cosine similarity |
| 1 Unsup. | Clustering | k-means VQ codec baseline |
| 1 Unsup. | Anomaly detection | Isolation Forest on patch uncertainty features |
| 1 Unsup. | Association rules, recommendation | Apriori on failure logs; mode recommender |
| 2 Sup. | kNN, linear/logistic regression, trees | Similar-image lookup, log-bpp prediction, patch risk, explainable mode rules |
| 3 Adv. | Ridge/Lasso, PCA, SVM, RF/XGBoost | Bpp/quality predictors, PCA codec, high-risk classifier, rate control |

## 6. Setup

Requires Python 3.10+ and PyTorch 2.x. A CUDA GPU (12 GB or more) is needed for training. Free options are Kaggle (T4/P100) and Colab; debug on CPU with tiny settings first. Total compute is roughly 120-180 GPU-hours.

```bash
conda create -n truesight python=3.10 -y
conda activate truesight

# PyTorch with CUDA: pick the command for your CUDA version at pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
python -c "import torch, compressai, torchac, diffusers; print(torch.cuda.is_available())"
```

Notes: `torchac` compiles a C++ extension on first use and needs `ninja` and a C++ compiler. If `facenet-pytorch` conflicts with your torch version, install it with `--no-deps`.

## 7. Evaluation protocol

**Test sets**

| Set | Content | Used for |
|---|---|---|
| Kodak (24 images) | Natural photos, 768x512 | RD curves, BD-rate, LPIPS, figures |
| CLIC 2020 test | Professional + mobile photos | LPIPS, FID on 256x256 crops (~2,000) |
| TextSet (300-500) | Signs, labels, documents | Text Faithfulness Rate |
| FaceSet (300-500) | Faces at various sizes | Face Identity Rate |

**Methods compared:** JPEG and WebP at matched bpp; PCA and VQ codecs; VAE codec alone; plain diffusion decode; TrueSight (diffusion + ROI + bin-consistency); TrueSight safe mode (flagged patches replaced by the coarse decode).

**Metrics:** bpp (header and ROI streams included), PSNR, MS-SSIM, LPIPS (AlexNet), FID (clean-fid), Text Faithfulness Rate (EasyOCR, CER <= 0.05), Face Identity Rate (InceptionResnetV1 vggface2, cosine >= 0.7), consistency rate, hallucination AUROC, decode time.

**Ablations**

| ID | Ablation |
|---|---|
| A1 | GMM K = 1 vs K = 3 |
| A2 | GDN vs ReLU in the codec |
| A3 | CFG scale w in {0, 1, 1.5, 2, 3} |
| A4 | With / without bin-consistency guidance |
| A5 | With / without ROI coding |
| A6 | DDIM steps {10, 25, 50} |
| A7 | Linear vs cosine noise schedule |
| A8 | Adam vs AdamW; constant vs cosine LR |

A small user study (at least 10 people, 20 pairs each) reports preference with a 95% binomial confidence interval and the rate at which people notice changes.

## 8. Success targets

| Metric | Target at the lowest mode (~0.03-0.06 bpp) |
|---|---|
| BD-rate of VAE codec vs JPEG (PSNR, Kodak) | At least 30% bit savings |
| LPIPS | At least 25% lower than the VAE codec alone at the same bpp |
| FID (CLIC 2020 crops) | Lower than JPEG, WebP and the VAE codec at the same bpp |
| Text Faithfulness Rate | At least 2x plain diffusion decoding |
| Face Identity Rate | At least 1.5x plain diffusion decoding |
| Hallucination detection AUROC | At least 0.75 |
| Rate-control accuracy | Within +/-10% of target size for 80% of images |
| Decode bit-exactness | 100% (same latent before and after the bitstream) |

Targets are not guarantees. If one is missed, record the real number and the reason.

## 9. Twelve-week plan

| Week | Theme | Key output |
|---|---|---|
| 1 | Setup, data, metrics | Clean dataset splits, tested metrics module |
| 2 | Classical and course baselines | RD plot: JPEG, WebP, PCA, VQ; first size predictor |
| 3 | Autoencoder and first VAE codec | AE results; one-lambda VAE trained |
| 4 | GMM hyperprior codec + bitstream | Full RD curve, bit-exact files, K ablation |
| 5 | Diffusion from scratch | Schedules, forward figure, unconditional samples |
| 6 | Conditional diffusion decoder + CFG | LPIPS gains, CFG sweep |
| 7 | Protected regions (ROI) | Text and face detection, ROI coding, first faithfulness numbers |
| 8 | Faithfulness guidance | Bin-consistency guidance tuned; main contribution results |
| 9 | Honesty layer | Hallucination detectors, AUROC table, uncertainty maps |
| 10 | Intelligence layer | Rate control, recommender, association rules |
| 11 | Full evaluation | All tables, ablations, user study |
| 12 | Demo, report, viva | CLI, API, UI, final report |

Each week ends with a done-criteria checklist. If a target is missed, log the real number and reason in `report/weekly_log.md`.

## 10. Usage (planned)

These commands will work once the corresponding modules are implemented.

```bash
python scripts/prepare_data.py                                  # download, clean, split datasets (Step 1)
python scripts/train_all_lambdas.py --no-ablation               # the six codecs (Step 7); resumes from last.pt, crash-safe
python -m truesight.diffusion.train_diffusion --codec experiments/step7_codec/lmbda0.0018/final.pt --out experiments/step10_diffusion
python scripts/eval_codec.py && python scripts/eval_diffusion.py
python scripts/make_figures.py                                  # every table and figure from the evaluation outputs
pytest tests/
```

Everything in one resumable command: `bash scripts/run_full_pipeline.sh` (see [CLOUD_RUNBOOK.md](CLOUD_RUNBOOK.md)).
On Kaggle, with 12-hour sessions that pause and continue by themselves: `python scripts/kaggle_run.py --sync hf:USER/REPO`
(see [KAGGLE.md](KAGGLE.md)). Training saves its exact state on a time limit or a crash and continues from there.

## 11. Conventions

- Images are PyTorch tensors of shape `(B, 3, H, W)` in `[0, 1]`; the diffusion model works in `[-1, 1]`.
- `bpp = file_bits / (H x W)`, including header and ROI streams.
- `data/raw/` is never edited. Each run in `experiments/` gets its own folder (config, `logs.csv`, checkpoints, figures).
- Checkpoints, datasets and `.tsi` files are git-ignored.

## References

Balle et al. 2018 (hyperprior VAE codec); Minnen et al. 2018; Cheng et al. 2020; JPEG AI (ISO/IEC 6048-1); HiFiC; CDC; PerCo; DiffEIC; Wang et al. 2025, *Beyond Hallucinations*.
