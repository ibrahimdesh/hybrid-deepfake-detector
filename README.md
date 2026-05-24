# Spatiotemporal Deepfake Detection with Cross-Manipulation Generalization

A Hybrid CNN–Transformer approach on FaceForensics++.

This repository contains the full experimental code for an in-depth study of
video-level deepfake detection. We train five models on FaceForensics++ (C23),
four established baselines and one proposed hybrid CNN–Transformer, and evaluate
them both in-domain and on a held-out manipulation type (FaceShifter) that no
model sees during training.

The study is framed as a scientific investigation, and its central finding is a
**negative result**: adding a temporal Transformer on top of a strong per-frame
CNN backbone does not improve detection or cross-manipulation generalization
under our data and compute budget. The value of the work is in explaining *why*,
not in claiming a better detector.

---

## Repository layout

```
.
├── utils.py                                  # Shared dataset, training loop, metrics, plotting
├── 01_preprocessing.ipynb                    # Videos → face-crop tensor clips
├── 03_baseline_xception.ipynb               # Baseline 1: Xception (frame-level CNN)
├── 04_baseline_efficientnet.ipynb           # Baseline 2: EfficientNet-B3 (frame-level CNN)
├── 05_baseline_cnn_lstm.ipynb               # Baseline 3: CNN + BiLSTM (recurrent temporal)
├── 06_baseline_vit.ipynb                    # Baseline 4: ViT-Base/16 (frame-level Transformer)
├── 07_proposed_hybrid_cnn_transformer.ipynb # Proposed model: Hybrid CNN–Transformer
├── 08_cross_manipulation_evaluation.ipynb   # Held-out FaceShifter generalization test
├── 09_grad_cam.ipynb                        # Grad-CAM interpretability visualizations
├── 10_results_aggregation.ipynb            # Collects all results into final tables/figures
├── figures/                                 # Generated plots used in the report
└── results/                                 # Per-model checkpoints, metrics, training history
```

Note on numbering: the notebooks are numbered to indicate execution order. There
is no notebook 02; the sequence intentionally jumps from `01` to `03`.

---

## Pipeline overview

Preprocessing runs once. The five model notebooks then run in any order (they all
consume the same preprocessed tensors), and the final three notebooks aggregate
and analyze the trained models.

```
        01 (preprocess, run once)
                  │
   ┌──────┬───────┼───────┬───────┐
   03     04      05      06      07          (train the 5 models, any order)
   │      │       │       │       │
   └──────┴───────┴───────┴───────┘
                  │
   ┌──────────────┼──────────────┐
   08             09             |            (evaluation + interpretability)
   └──────────────┼──────────────┘
                  │
                 10                           (aggregate everything into report tables)
```

---

## Notebooks, in running order

Run in the order below. Steps 3–7 depend only on step 1 and can run in any order
among themselves. Steps 8–10 depend on the trained models from steps 3–7.

1. **`01_preprocessing.ipynb`** — run once, first. Converts raw FaceForensics++
   videos into fixed-size face-crop tensor clips so the model notebooks never
   repeat this expensive step. Samples 24 evenly spaced frames per video, detects
   the largest face with MTCNN, crops with a 30% margin, resizes to 224×224, and
   saves each clip as a half-precision tensor of shape `[24, 3, 224, 224]`. Also
   writes `splits.json` (the 80/10/10 train/val/test split over the five training
   classes, with FaceShifter kept as a separate held-out group). **Output:**
   preprocessed tensors and `splits.json`, read by every later notebook.

2. **`03_baseline_xception.ipynb`** — Baseline 1. Trains Xception, the standard
   frame-level CNN baseline from the original FaceForensics++ paper. Runs on each
   of the 24 frames and averages the per-frame logits for a clip-level score.
   Pure-CNN, spatial-only.

3. **`04_baseline_efficientnet.ipynb`** — Baseline 2. Trains EfficientNet-B3, a
   stronger, more parameter-efficient frame-level CNN (same frame-averaging
   setup). This is also the backbone reused inside the proposed hybrid, so the two
   form a controlled ablation of the temporal head. Pure-CNN, spatial-only.

4. **`05_baseline_cnn_lstm.ipynb`** — Baseline 3. Trains a ResNet-18 backbone
   whose per-frame features feed a two-layer bidirectional LSTM. The recurrent
   temporal baseline: unlike the frame-averaging CNNs, it models the time
   dimension. Pure-CNN, spatiotemporal (recurrent).

5. **`06_baseline_vit.ipynb`** — Baseline 4. Trains a Vision Transformer
   (ViT-Base/16), which processes image patches with self-attention rather than
   convolutions (same frame-averaging setup). A pure-Transformer, spatial-only
   point of comparison.

6. **`07_proposed_hybrid_cnn_transformer.ipynb`** — Proposed model. Trains the
   project's main contribution: an EfficientNet-B3 backbone whose 24 per-frame
   features are aggregated by a four-layer Transformer encoder with a learnable
   CLS token and learned positional embeddings, followed by a small MLP head.
   Trained end-to-end with no frozen layers. Produces a single clip-level decision
   directly. CNN-plus-Transformer, spatiotemporal (attention).

7. **`08_cross_manipulation_evaluation.ipynb`** — Generalization test. Loads all
   five trained checkpoints and evaluates them on the held-out FaceShifter
   manipulation. Because FaceShifter clips are all fake, real clips from the
   in-domain test split are mixed in so AUC is well-defined. Reports in-domain vs.
   FaceShifter AUC and the AUC drop — the central experiment. **Depends on steps 2–6.**

8. **`09_grad_cam.ipynb`** — Interpretability. Generates Grad-CAM heatmaps showing
   which facial regions drive each model's decision, on nine fixed clips (three
   real, three in-domain fake, three FaceShifter). Uses the last convolutional
   block for the CNN backbones and the final attention block (with patch-grid
   reshape) for ViT. **Depends on steps 2–6.**

9. **`10_results_aggregation.ipynb`** — Final tables and figures. Pulls together
   every result file produced by notebooks 03–09 and assembles the in-domain
   table, per-manipulation table, cross-manipulation table, combined summary
   table, and the comparison figures used in the report. **Depends on steps 2–8.**

---

## Shared utilities (`utils.py`)

All notebooks import from `utils.py`, which centralizes the components that must
be identical across models for a fair comparison:

- `DeepfakeClipDataset` — loads preprocessed clips, applies ImageNet
  normalization, and applies train-time augmentation (horizontal flip and
  brightness jitter, applied consistently across all 24 frames).
- `make_balanced_sampler` — a `WeightedRandomSampler` with inverse-frequency
  weights to counter the roughly 4:1 fake:real class imbalance.
- `compute_metrics` — AUC, accuracy, precision, recall, F1, and confusion matrix.
- `train_one_epoch` / `evaluate` / `run_training` — the shared training loop with
  mixed-precision training, cosine-annealing schedule, early stopping on
  validation AUC, and best-checkpoint saving.
- `EarlyStopping`, checkpoint helpers, and `plot_training_curves`.

The random seed is fixed to 42 and cuDNN autotuning is enabled, so runs are
reproducible up to the small nondeterminism introduced by autotuned kernels.

---

## Training configuration (identical across all five models)

To ensure differences reflect architecture rather than tuning, every model uses:

- **Loss:** binary cross-entropy on the clip-level logit (`BCEWithLogitsLoss`)
- **Optimizer:** AdamW, learning rate 1e-4, weight decay 1e-4
- **Schedule:** cosine annealing over 25 epochs; early stopping (patience 5) on
  validation AUC; best checkpoint by validation AUC
- **Precision:** fp16 autocast with gradient scaling
- **Batch size:** set per model by VRAM (8 / 8 / 16 / 4 / 4 for
  Xception / EfficientNet / CNN+BiLSTM / ViT / hybrid)
- **Augmentation:** horizontal flip (p=0.5), brightness jitter (±10%),
  ImageNet normalization

Only memory-dictated settings (batch size) and architecture-intrinsic settings
(LSTM hidden size, Transformer depth/width) differ between models. No per-model
hyperparameter sweep was performed; this is a deliberate fair-comparison choice
and is discussed as a limitation in the report.

---

## Results summary

In-domain (5 trained classes, shared test split) and cross-manipulation
(held-out FaceShifter) AUC:

| Model | Params (M) | In-domain AUC | FaceShifter AUC | AUC drop |
|---|---|---|---|---|
| Xception | 20.8 | 0.9944 | 0.7015 | +0.2929 |
| EfficientNet-B3 | 10.7 | 0.9976 | 0.7343 | +0.2633 |
| CNN+BiLSTM | 13.7 | 0.9805 | 0.7591 | +0.2214 |
| ViT-Base/16 | 85.8 | 0.7066 | 0.6880 | +0.0186 |
| Hybrid (ours) | 74.9 | 0.9458 | 0.6273 | +0.3185 |

EfficientNet-B3 is the strongest in-domain model; the proposed hybrid scores
below its own backbone; every model degrades sharply on the unseen manipulation;
and the hybrid generalizes worst of all.

---

## Dataset

FaceForensics++ at the C23 (high-quality) compression level.

- **Training classes:** real `original` videos plus four manipulations —
  Deepfakes, Face2Face, FaceSwap, NeuralTextures (700 videos per class).
- **Held out for cross-manipulation testing:** FaceShifter (400 videos), never
  seen during training.

Obtain FaceForensics++ from its official source and comply with its research
license and access terms before use. The dataset is not included in this
repository.

---

## Environment

- Python 3.12+, PyTorch 2.11, CUDA 12.8
- Developed on a single NVIDIA RTX 5080 (16 GB), Ryzen 7 9800X3D, 32 GB RAM

A `requirements.txt` is recommended for reproducible setup. Core dependencies:
`torch`, `torchvision`, `timm`, `facenet-pytorch` (MTCNN), `scikit-learn`,
`numpy`, `matplotlib`, `tqdm`, `tensorboard`, and a Grad-CAM implementation.

---

## How to reproduce

1. Obtain FaceForensics++ (C23) and set the dataset path in `01_preprocessing.ipynb`.
2. Run `01_preprocessing.ipynb` once to generate the tensor clips and `splits.json`.
3. Run `03`–`07` to train the five models (any order). Checkpoints and metrics are
   written under `results/`.
4. Run `08` and `09` for the cross-manipulation evaluation and Grad-CAM plots.
5. Run `10` to assemble the final tables and figures.

Trained checkpoints are large and are hosted separately rather than committed to
this repository; see `results/` for the expected per-model layout.
