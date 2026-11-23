"""The skeleton of the final report (Week 12), with every table filled in from the measured results.

What is generated and what is not:
  * generated, from files that exist: the course-mapping table (each row names the code file or experiment that proves the topic and
    says whether it exists), the results tables (success targets, ablations A1-A8, faithfulness, user study), the list of deviations
    from the guide, the limitations that follow from what was not measured, and the provenance of the numbers;
  * NOT generated: the narrative chapters (problem, solution, method, discussion, future work, startup path). They are marked
    ``TODO(author)`` with the question each chapter has to answer. The report is yours to write; this only removes the copying of
    numbers, so every figure in it comes from a script and cannot drift from the experiments.
"""
import json
from pathlib import Path

from . import report as R

# course, topic, how it is used, evidence (files that must exist / experiment names)
COURSE_MAP = [
    ("5 GenAI", "MLE", "Entropy model trained by maximum likelihood: rate = -log2 p(y_hat)", ["truesight/codec/entropy_gmm.py"]),
    ("5 GenAI", "GMM", "Discretised Gaussian mixture (K=3) per latent element; ablation A1", ["truesight/codec/vae_codec.py"]),
    ("5 GenAI", "Autoencoders", "Plain AE baseline without rate control (Step 6)", ["truesight/codec/layers.py"]),
    ("5 GenAI", "VA / VAE", "Codec = VAE with uniform-noise posterior, learned prior, rate + lambda * distortion", ["truesight/codec/vae_codec.py"]),
    ("5 GenAI", "Diffusion: forward", "Own noise schedules and q(x_t | x_0), visualised", ["truesight/diffusion/schedule.py", "truesight/diffusion/uncond.py"]),
    ("5 GenAI", "Diffusion: backward", "eps-prediction U-Net, DDPM training, DDIM sampling written from scratch", ["truesight/diffusion/sample.py"]),
    ("5 GenAI", "Conditioning", "Decoder conditioned on the coarse reconstruction by channel concatenation", ["truesight/diffusion/train_diffusion.py"]),
    ("5 GenAI", "Guiding", "Classifier-free guidance, bin-consistency guidance, ROI-lock guidance (A3, A4)", ["truesight/diffusion/guidance.py"]),
    ("4 NN", "MLP, perceptron", "Perceptron and MLP detectors of patch hallucination risk", ["truesight/analysis/risk_model.py"]),
    ("4 NN", "Activations", "GDN vs ReLU in the codec (A2)", ["truesight/codec/vae_codec.py"]),
    ("4 NN", "Gradient descent, optimizers, schedules", "Adam vs AdamW, cosine vs constant LR, warm-up, clipping, aux optimizer (A8)", ["truesight/diffusion/ablation.py"]),
    ("4 NN", "Training deep nets, CNN", "Conv encoder / decoder, U-Net, mixed precision, EMA", ["truesight/diffusion/train_diffusion.py"]),
    ("1 Unsup.", "Data foundations", "Collection, cleaning, pHash de-duplication, crops, colour handling", ["truesight/data.py", "truesight/data_prepare.py"]),
    ("1 Unsup.", "Distance measures", "MSE / PSNR, MS-SSIM, LPIPS, CER, cosine similarity of face embeddings", ["truesight/metrics.py"]),
    ("1 Unsup.", "Clustering", "k-means vector-quantisation codec baseline", ["truesight/baselines/vq_codec.py"]),
    ("1 Unsup.", "Anomaly detection", "Isolation Forest on patch uncertainty features", ["truesight/analysis/risk_model.py"]),
    ("1 Unsup.", "Association rules", "Apriori on failure logs -> product rules", ["truesight/analysis/rules.py"]),
    ("1 Unsup.", "Recommendation", "Mode recommender from content, network and user history", ["truesight/analysis/recommender.py"]),
    ("2 Sup.", "kNN", "Similar-image lookup for recommendations; nearest codeword in VQ", ["truesight/analysis/recommender.py"]),
    ("2 Sup.", "Linear regression", "log-bpp from image complexity features (Steps 5 and 14)", ["truesight/analysis/rate_control.py", "scripts/train_size_predictor.py"]),
    ("2 Sup.", "Logistic regression", "Patch-level hallucination risk used to add protected regions", ["truesight/analysis/risk_model.py"]),
    ("2 Sup.", "Decision trees", "Explainable mode-selection rules", ["truesight/analysis/recommender.py"]),
    ("3 Adv.", "Function approximation, loss, hyperparameters", "Rate-distortion loss; lambda, K, guidance scale as hyperparameters", ["truesight/codec/vae_codec.py"]),
    ("3 Adv.", "Bias-variance, regularisation", "Learning curves of the risk models", ["truesight/analysis/risk_model.py"]),
    ("3 Adv.", "Ridge, Lasso, Elastic Net", "bpp and quality predictors; Lasso feature selection", ["truesight/analysis/rate_control.py"]),
    ("3 Adv.", "PCA", "PCA (Karhunen-Loeve) transform-coding baseline; feature visualisation", ["truesight/baselines/pca_codec.py"]),
    ("3 Adv.", "SVM", "RBF-SVM hallucination detector", ["truesight/analysis/risk_model.py"]),
    ("3 Adv.", "Random Forest, XGBoost", "Rate control: bpp and LPIPS per mode", ["truesight/analysis/rate_control.py"]),
]

# differences between the code and the guide, kept here so the report states them instead of hiding them
DEVIATIONS = [
    "The file format is `.sphota` with magic `SPH1`; the guide names it `.tsi` / `TSI1`. The fields and sizes are the same.",
    "Guidance uses gradient normalisation and a per-step cap of 0.1 by default; `normalize=False, max_step=None` is the literal "
    "step of the guide.",
    "The codec clamps the log-scale before exp() (a numerical safeguard) and builds the arithmetic-coder CDFs in channel chunks.",
    "The PCA baseline codes luma and half-resolution chroma in YCbCr, as JPEG does.",
    "The Step 14 six-mode table holds the LPIPS of the coarse decode; text / face pass-fail is only recorded for the four recommender "
    "modes.",
    "Rate control chooses the highest-quality mode whose predicted size fits the target (the guide's wording is ambiguous with six "
    "discrete modes).",
    "The CLIC 2020 test labels are not public: the CLIC valid lists stand in for the test split.",
    "Evaluation images larger than `--max-pixels` (default 768 x 512) are replaced by the window that covers the most text / faces, "
    "because guided decoding needs the whole image at once and the U-Net's cost grows with the square of the image size.",
]

TODO_CHAPTERS = [
    ("1. Problem", "Why images cost too much to store and send; why classical codecs fail below ~0.1 bpp; why generative decoders "
                   "hallucinate and why that matters (guide sections 2.1-2.5)."),
    ("2. Solution", "The five building blocks and what is new: faithfulness guidance in the bin set of the transmitted integers, "
                    "protected regions, the honesty layer, the intelligence layer (guide sections 3.1-3.3)."),
    ("3. Method", "The codec as a VAE (derive the rate as a negative log-likelihood and the loss as a weighted ELBO), the bitstream, the "
                  "diffusion decoder, the guidance losses, the detectors and the mode recommender. One equation or figure per component."),
    ("4. Experiments", "Datasets and splits (use the dataset report), training settings and compute actually used, metrics, baselines."),
    ("5. Discussion", "Where the targets were met or missed and why (use the table above and `report/weekly_log.md`); what the ablations "
                      "show; failure cases from the guidance examples and the association rules."),
    ("6. Future work and startup path", "Video, on-device decoding, the licensing / SDK path of guide section 13."),
]


def evidence_status(root, files):
    root = Path(root)
    return all((root / f).exists() for f in files)


def course_table(root):
    lines = ["| Course | Topic | Where it is used | Evidence | Exists |", "|---|---|---|---|---|"]
    for course, topic, use, files in COURSE_MAP:
        lines.append(f"| {course} | {topic} | {use} | {', '.join(f'`{f}`' for f in files)} | "
                     f"{'yes' if evidence_status(root, files) else 'MISSING'} |")
    return "\n".join(lines) + "\n"


def limitations(targets, ablations):
    """Honest limitation bullets derived from the measured state."""
    out = []
    for _, r in targets.iterrows():
        flag = R._as_flag(r["met"])
        if flag is None:
            out.append(f"{r['target']}: not measured ({r['note'] or 'evaluation output not found'}).")
        elif flag is False:
            out.append(f"{r['target']}: NOT met (measured {r['measured']}, required {r['requirement']}). {r['note']}".strip())
    out += [f"Ablation {r['id']} ({r['ablation']}) was not run." for _, r in ablations.iterrows() if r["status"] != "done"]
    return out or ["Every target was measured and met; every ablation was run."]


def build_report(root, results_dir, study_md=None):
    """Markdown text of the report skeleton. ``results_dir`` is report/results (written by scripts/make_figures.py)."""
    root, results_dir = Path(root), Path(results_dir)

    def read(name):
        p = results_dir / name
        return p.read_text(encoding="utf-8") if p.exists() else "_not generated yet: run scripts/make_figures.py_\n"

    prov = {}
    if (results_dir / "provenance.json").exists():
        prov = json.loads((results_dir / "provenance.json").read_text())
    targets = R.load_csv(results_dir / "summary_table.csv")
    ablations = R.load_csv(results_dir / "ablation_table.csv")
    lim = limitations(targets, ablations) if targets is not None and ablations is not None else \
        ["Results were not generated yet, so no limitation could be derived."]
    parts = ["# TrueSight / Sphota: a faithful generative image codec\n",
             "> Generated skeleton: the tables below come from `report/results/`; chapters marked TODO(author) are for you to write.\n",
             f"Numbers generated from commit `{str(prov.get('commit'))[:10]}` "
             f"({'frozen: tag eval-freeze on a clean tree' if prov.get('frozen') else 'NOT frozen: tag eval-freeze before quoting them'}), "
             f"{prov.get('generated', '')}.\n"]
    for title, hint in TODO_CHAPTERS[:4]:
        parts.append(f"## {title}\n\nTODO(author): {hint}\n")
    parts.append("### Course mapping\n\n" + course_table(root))
    parts.append("## Results\n\n### Success targets\n\n" + read("summary_table.md"))
    parts.append("### Ablations\n\n" + read("ablation_table.md"))
    parts.append("### Faithfulness and detectors\n\n" + read("faithfulness.md"))
    parts.append("### User study\n\n" + (study_md or read("user_study.md")))
    parts.append(f"## {TODO_CHAPTERS[4][0]}\n\nTODO(author): {TODO_CHAPTERS[4][1]}\n")
    parts.append("## Limitations\n\n" + "\n".join(f"- {x}" for x in lim) + "\n\n### Differences from the build guide\n\n" +
                 "\n".join(f"- {x}" for x in DEVIATIONS) + "\n")
    parts.append(f"## {TODO_CHAPTERS[5][0]}\n\nTODO(author): {TODO_CHAPTERS[5][1]}\n")
    return "\n".join(parts)
