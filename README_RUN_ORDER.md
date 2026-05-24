# Project Run Order and Instructions

This file is your map. Read it once, then refer back whenever you're unsure what's next.

---

## Workflow at a glance

You're training on a local desktop (RTX 5080, Windows). The Mac M3 is for writing the report and slides. Kaggle is a backup option only.

| # | Notebook | What it does | Approx. time |
|---|---|---|---|
| 0a | `00b_environment_setup.ipynb` | Verifies CUDA / PyTorch / GPU works for your RTX 5080. **Run this first, every time you set up the venv.** | 1 min |
| 0b | `00_dataset_exploration.ipynb` | Sanity-checks the dataset structure and tries opening one video. | 2–3 min |
| 1 | `01_preprocessing.ipynb` | The big one-time job. Videos → face crop tensors. | 15–20 min dry run, 45–90 min full run |

More notebooks will be added in the next phase (models, training, evaluation).

---

## One-time machine setup

Do this in **PowerShell** on the lab desktop. You only do it once.

### 1. Install Python and Git

- Get Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/windows/). Tick *Add Python to PATH* during install.
- Get Git from [git-scm.com](https://git-scm.com/download/win) (optional but useful for version-controlling your code).

### 2. Create a project folder

```powershell
mkdir C:\Users\<you>\Documents\deepfake_project
cd C:\Users\<you>\Documents\deepfake_project
```

Put all the `.ipynb` files into this folder.

### 3. Create and activate a venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your prompt now. If you get a script-execution error, run PowerShell as Administrator once and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Install JupyterLab and PyTorch

```powershell
python -m pip install --upgrade pip
pip install jupyterlab
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

The `--index-url` line is the critical one. Without it, you get CPU-only PyTorch that won't see your RTX 5080.

### 5. Install the rest of the project dependencies

```powershell
pip install decord facenet-pytorch timm tqdm matplotlib pandas scikit-learn opencv-python pillow kaggle
```

### 6. Launch JupyterLab

```powershell
jupyter lab
```

A browser tab opens. From now on, the only command you need each session is `.\.venv\Scripts\Activate.ps1` then `jupyter lab`.

---

## Step 1 — Verify the environment (`00b_environment_setup.ipynb`)

Open it in JupyterLab. Run all cells. **Every check must pass with a ✓.**

The most critical check is **#5 — Compute capability**. If your PyTorch doesn't list `sm_120`, your GPU won't actually be used during training. The notebook tells you exactly how to fix it (reinstall with the `cu128` index URL).

Don't skip this notebook. It catches the most common failure mode for RTX 5080 users.

---

## Step 2 — Download the dataset

In PowerShell (venv still active):

```powershell
# Get a Kaggle API token first: kaggle.com/settings/account → Create New Token.
# Move the downloaded kaggle.json to C:\Users\<you>\.kaggle\kaggle.json.

kaggle datasets download xdxd003/ff-c23 -p D:\datasets\ff-c23 --unzip
```

This downloads 17.92 GB. Pick a drive with at least 25 GB free. The path `D:\datasets\ff-c23\` is what the notebooks expect by default — if you put it somewhere else, update the `LOCAL_DATASET_ROOT` paths in the notebooks.

Alternative: download manually from [the dataset page](https://www.kaggle.com/datasets/xdxd003/ff-c23) in a browser, then extract.

---

## Step 3 — Explore the dataset (`00_dataset_exploration.ipynb`)

Open in JupyterLab. Run all cells. Confirm:

- All 6 manipulation folders + `original` + `csv` are present
- About 1000 videos per class
- The `csv/` folder contains split files we can read
- The sample video opens and displays a face

If any of these fail, fix before going further.

---

## Step 4 — Dry run preprocessing (`01_preprocessing.ipynb`)

1. Open the notebook.
2. **Leave `VIDEOS_PER_CLASS = 50`** in the config cell.
3. Run all cells.
4. Total time: about 15–20 minutes on RTX 5080.
5. At the end, check the visual sanity check: face crops should be properly centered, not cut off, not capturing the background.

If something looks wrong (e.g., bounding boxes are off), stop here and tell Claude in the next session what you saw.

---

## Step 5 — Full preprocessing run

1. In the same notebook, change the config cell:
   ```python
   VIDEOS_PER_CLASS = 700
   ```
2. Re-run all cells.
3. Total time: 45–90 minutes on RTX 5080.

The script is **resume-safe** — it skips any video that already has a `.pt` file saved. So if your computer reboots or the kernel dies, just rerun.

At the end, you'll have:
- 5 training classes × ~700 tensors each = ~3500 tensors
- 1 held-out class (FaceShifter) × ~400 tensors
- 1 `splits.json` file
- All in `D:\datasets\ff-c23-preprocessed\`

This is the input every training notebook from now on will consume.

---

## Tips and gotchas

- **Always activate the venv first.** Every PowerShell session: `.\.venv\Scripts\Activate.ps1`.
- **The RTX 5080 needs PyTorch 2.7+ with CUDA 12.8 wheels.** Older PyTorch silently falls back to CPU. The environment notebook catches this.
- **Don't change `FRAMES_PER_CLIP` or `FACE_CROP_SIZE` mid-project.** If you do, you have to redo preprocessing.
- **Failed videos are normal.** A handful of corrupted videos exist in every dataset. As long as >95% succeed, you're fine.
- **Kaggle is for backup only now.** If your local GPU is busy, Kaggle accounts are still useful for running ablations in parallel. The notebooks auto-detect both environments.

---

## What's coming next (not yet built)

In the next batch of notebooks, Claude will produce:

| Notebook | Purpose |
|---|---|
| `02_dataloader_and_utils.ipynb` | Shared dataloader + utility functions |
| `03_baseline_xception.ipynb` | Baseline 1: Xception (FF++ standard) |
| `04_baseline_efficientnet.ipynb` | Baseline 2: EfficientNet-B3 |
| `05_baseline_cnn_lstm.ipynb` | Baseline 3: CNN + LSTM (temporal) |
| `06_baseline_vit.ipynb` | Baseline 4: ViT-Base/16 |
| `07_proposed_hybrid_model.ipynb` | The proposed contribution: EfficientNet-B3 + Transformer |
| `08_ablations.ipynb` | Variants: w/o temporal, w/o spatial, different depths, FFT branch |
| `09_evaluation_and_generalization.ipynb` | Final test-set numbers + FaceShifter cross-manipulation test |
| `10_gradcam_visualization.ipynb` | Heatmaps for the presentation |

When you're ready to start Phase 2, use a handoff prompt similar to the one Claude gave you before — the next session needs to know the project plan, the local desktop environment, and the path to the preprocessed dataset.

---

## Final submission packaging (for May 24)

Your submission `.zip` will contain:
- The final report (PDF)
- All `.ipynb` files (with the *How to Run* cells deleted)
- A shorter README explaining the run order
- Trained model checkpoints (or download links if too large)

Don't worry about this yet. We package it on the last day.
