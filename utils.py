"""Shared utilities for the deepfake detection project.

"""

# Import necessary libraries
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


# Set the seed value
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Enable cuDNN auto-tuner to select the fastest convolution algorithms
torch.backends.cudnn.benchmark = True


# Set the path for the pre-processed tensors
PREPROCESSED_TENSORS_DIR_PATH = Path("dataset/ff-c23-preprocessed")

# Set the path to store the results
RESULTS_DIR_PATH = Path("./results")
RESULTS_DIR_PATH.mkdir(exist_ok=True)

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------- DATASET CLASS  --------------------------------
class DeepfakeClipDataset(Dataset):
    """
    This function loads preprocessed face-croped tensor clips for deepfake classification.
    
    Each item returns:
        - clip: tensor of shape [T, 3, H, W], normalized with ImageNet stats
        - label: 0 (real) or 1 (fake)
        - metadata: dict with class_name and original tensor path
    """
    
    def __init__(self, splits_json_path, split_name, preprocessed_tensors_dir_path, training_augment=False):
        with open(splits_json_path, "r") as f:
            all_splits = json.load(f)
        
        if split_name not in all_splits:
            raise ValueError(f"Split '{split_name}' not in splits.json. Available: {list(all_splits.keys())}")
        
        self.samples = all_splits[split_name]
        self.preprocessed_tensors_dir_path = Path(preprocessed_tensors_dir_path)
        self.training_augment = training_augment
        
        # Precompute normalization tensors (broadcast over [T, 3, H, W]).
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        tensor_path = self.preprocessed_tensors_dir_path / sample["tensor_path"]
        
        # Converet the stored tensor from float16 to float32 in [0, 1]
        clip = torch.load(tensor_path, weights_only=True).float()  # [T, 3, H, W]
        
        # Light training-time augmentation. We'll apply consistently across all frames of the clip
        if self.training_augment:
            # Apply Horizontal flip to the 50% of frames
            if random.random() < 0.5:
                clip = torch.flip(clip, dims=[3])
            # Apply brightness jitter (a small random scaling of pixel values)
            brightness_factor = 1.0 + (random.random() - 0.5) * 0.2  # ±10%
            clip = (clip * brightness_factor).clamp(0, 1)
        
        # ImageNet normalization.
        clip = (clip - self.mean) / self.std
        
        label = sample["label"]
        metadata = {"class_name": sample["class_name"], "tensor_path": sample["tensor_path"]}
        return clip, label, metadata
    


# ------------------ CLASS BALANCE SAMPLER ----------------------------------
def make_balanced_sampler(dataset):
    """
    This function returns a WeightedRandomSampler that generates balanced real/fake batches of the dataset.
    """
    labels = np.array([sample["label"] for sample in dataset.samples])
    class_counts = np.bincount(labels)
    # Weight for each sample = 1 / (count of its class).
    sample_weights = 1.0 / class_counts[labels]
    
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ----------------------- EVALUATION METRICS ------------------------------
def compute_metrics(true_labels, predicted_probs, threshold=0.5):
    """
    This function computes the classification metrics. And returns the results as a dict.
    """
    true_labels = np.asarray(true_labels)
    predicted_probs = np.asarray(predicted_probs)
    predicted_labels = (predicted_probs >= threshold).astype(int)
    
    # AUC requires both classes present.
    if len(np.unique(true_labels)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(true_labels, predicted_probs)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="binary", zero_division=0
    )
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    
    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


# -------------------- TRAINING & EVALUATION LOOP -------------------
def train_one_epoch(model, dataloader, optimizer, loss_function, device, scaler=None):
    """
    This function runs one training epoch. Then, returns average loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="train", leave=False)
    for clips, labels, _ in progress_bar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass on GPU
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(clips).squeeze(-1)
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(clips).squeeze(-1)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, loss_function, device):
    """
    This function runs validation/test on the given model. 
    Returns (avg_loss, metrics_dict, per_sample_predictions).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_true_labels = []
    all_predicted_probs = []
    all_class_names = []
    
    for clips, labels, metadata in tqdm(dataloader, desc="eval", leave=False):
        clips = clips.to(device, non_blocking=True)
        labels_float = labels.to(device, non_blocking=True).float()
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(clips).squeeze(-1)
            loss = loss_function(logits, labels_float)
        
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_true_labels.extend(labels.numpy().tolist())
        all_predicted_probs.extend(probs.tolist())
        all_class_names.extend(metadata["class_name"])
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    metrics = compute_metrics(all_true_labels, all_predicted_probs)
    
    # Keep the per-sample predictions for later analysis (e.g., per-class breakdown).
    per_sample = {
        "true_labels": all_true_labels,
        "predicted_probs": all_predicted_probs,
        "class_names": all_class_names,
    }
    return avg_loss, metrics, per_sample


# ------------------------- EARLY STOPPING ---------------
class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode  # "max" for AUC/acc, "min" for loss
        self.best_score = None
        self.epochs_without_improvement = 0
        self.should_stop = False
    
    def step(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False  # improved (first epoch)
        
        improved = (current_score > self.best_score) if self.mode == "max" else (current_score < self.best_score)
        
        if improved:
            self.best_score = current_score
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
            return False


# -------------------------- CHECKPOINTS HELPERS ----------------
def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """
    This function saves thew best model state + optimizer + metadata.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, save_path)


def load_checkpoint(model, checkpoint_path, device, optimizer=None):
    """
    This function loads a saved checkpoint into model (and optionally optimizer).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

# ----------------------- FULL TRAINING LOOP ---------
def run_training(
    model,
    train_loader,
    validation_loader,
    optimizer,
    scheduler,
    loss_function,
    device,
    model_name,
    number_of_epochs,
    early_stopping_patience=5,
    use_mixed_precision=True,
    ):
    """
    This is full training loop that trains a model from start to finish. 
    Returns the path to the best checkpoint.
    """
    results_dir_path = RESULTS_DIR_PATH / model_name
    results_dir_path.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = results_dir_path / "best_model.pt"
    history_path = results_dir_path / "training_history.json"
    tensorboard_dir = results_dir_path / "tensorboard"
    
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    scaler = torch.amp.GradScaler("cuda") if use_mixed_precision else None
    early_stopper = EarlyStopping(patience=early_stopping_patience, mode="max")
    
    training_history = []
    best_validation_auc = -1.0
    
    print(f"\nTraining: {model_name}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"  Run `tensorboard --logdir {RESULTS_DIR_PATH.resolve()}` to view live curves.\n")
    
    for epoch in range(1, number_of_epochs + 1):
        epoch_start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        validation_loss, validation_metrics, _ = evaluate(model, validation_loader, loss_function, device)
        
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - epoch_start_time
        
        # TensorBoard scalars.
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/validation", validation_loss, epoch)
        writer.add_scalar("metrics/validation_auc", validation_metrics["auc"], epoch)
        writer.add_scalar("metrics/validation_accuracy", validation_metrics["accuracy"], epoch)
        writer.add_scalar("metrics/validation_f1", validation_metrics["f1"], epoch)
        writer.add_scalar("learning_rate", current_lr, epoch)
        
        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "lr": current_lr,
            "epoch_seconds": epoch_seconds,
        }
        training_history.append(history_entry)
        
        # Save best checkpoint by validation AUC
        improved = early_stopper.step(validation_metrics["auc"])
        if improved:
            best_validation_auc = validation_metrics["auc"]
            save_checkpoint(model, optimizer, epoch, validation_metrics, best_checkpoint_path)
        
        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:2d}/{number_of_epochs}  "
            f"train_loss={train_loss:.4f}  validation_loss={validation_loss:.4f}  "
            f"validation_auc={validation_metrics["auc"]:.4f}  validation_acc={validation_metrics["accuracy"]:.4f}  "
            f"({epoch_seconds:.1f}s){marker}"
        )
        
        # Persist history every epoch so we never lose it
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)
        
        if early_stopper.should_stop:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs).")
            break
    
    writer.close()
    print(f"\nTraining done. Best vvalidation AUC: {best_validation_auc:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    return best_checkpoint_path


# --------------------- PLOTTING HELPERS --------------------
def plot_training_curves(history_json_path, save_path=None):
    """
    This function plots loss + AUC + accuracy curves from a training_history.json file.
    """
    with open(history_json_path, "r") as f:
        history = json.load(f)
    
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    validation_losses = [h["validation_loss"] for h in history]
    validation_aucs = [h["validation_metrics"]["auc"] for h in history]
    validation_accs = [h["validation_metrics"]["accuracy"] for h in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(epochs, train_losses, label="train")
    axes[0].plot(epochs, validation_losses, label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(epochs, validation_aucs, color="C2")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Validation AUC")
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(epochs, validation_accs, color="C3")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Validation Accuracy")
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
