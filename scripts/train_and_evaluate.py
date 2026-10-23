import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Reproducibility & Hardware Settings
# ==========================================
SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(SEED)
torch.set_num_threads(4) # Utilize 4 CPU threads for optimal efficiency
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing on device: {device} | PyTorch version: {torch.__version__} | Python: {sys.version.split()[0]}")

os.makedirs('outputs/model', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
print("\n--- Step 1: Loading Datasets ---")
train_raw = pd.read_csv('train.csv')
dig_raw = pd.read_csv('Dig-MNIST.csv')
test_raw = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print(f"Raw train shape: {train_raw.shape}")
print(f"Raw Dig-MNIST shape: {dig_raw.shape}")
print(f"Raw test shape: {test_raw.shape}")

# Stratified Split of train.csv into Train (80%) and Held-out Validation (20%)
# 60,000 samples -> 48,000 train (4,800/class) and 12,000 val (1,200/class)
X_all = train_raw.iloc[:, 1:].values.astype(np.float32) / 255.0
y_all = train_raw['label'].values.astype(np.int64)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(sss.split(X_all, y_all))

np.savez('outputs/reports/train_val_split_indices.npz', train_idx=train_idx, val_idx=val_idx)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]

# Dig-MNIST independent evaluation set
X_dig = dig_raw.iloc[:, 1:].values.astype(np.float32) / 255.0
y_dig = dig_raw['label'].values.astype(np.int64)

# Test set (unlabeled)
test_ids = test_raw['id'].values
X_test = test_raw.iloc[:, 1:].values.astype(np.float32) / 255.0

print(f"Training set:   {X_train.shape[0]} samples (4,800 per class)")
print(f"Validation set: {X_val.shape[0]} samples (1,200 per class)")
print(f"Dig-MNIST set:  {X_dig.shape[0]} samples (1,024 per class)")
print(f"Test set:       {X_test.shape[0]} samples")

# Reshape to (N, 1, 28, 28)
X_train = X_train.reshape(-1, 1, 28, 28)
X_val = X_val.reshape(-1, 1, 28, 28)
X_dig = X_dig.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# Custom PyTorch Datasets
class KannadaDataset(Dataset):
    def __init__(self, images, labels=None, augment=False):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.augment:
            img = self._apply_label_preserving_aug(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img

    def _apply_label_preserving_aug(self, img):
        # img is (1, 28, 28)
        # Random small translation (+/- 2 pixels, ~7%) and subtle rotation (+/- 8 degrees)
        # NO horizontal or vertical flipping because Kannada numerals are orientation-critical
        angle = random.uniform(-8.0, 8.0) * (3.14159265 / 180.0)
        tx = random.uniform(-2.0, 2.0) / 14.0 # relative to [-1, 1]
        ty = random.uniform(-2.0, 2.0) / 14.0
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        theta = torch.tensor([
            [cos_a, -sin_a, tx],
            [sin_a,  cos_a, ty]
        ], dtype=torch.float32).unsqueeze(0)
        
        grid = F.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)
        img_aug = F.grid_sample(img.unsqueeze(0), grid, align_corners=False, mode='bilinear', padding_mode='zeros')
        return img_aug.squeeze(0)

# DataLoaders
BATCH_SIZE = 256
train_loader_base = DataLoader(KannadaDataset(X_train, y_train, augment=False), batch_size=BATCH_SIZE, shuffle=True)
train_loader_aug  = DataLoader(KannadaDataset(X_train, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True)
val_loader        = DataLoader(KannadaDataset(X_val, y_val, augment=False), batch_size=BATCH_SIZE, shuffle=False)
dig_loader        = DataLoader(KannadaDataset(X_dig, y_dig, augment=False), batch_size=BATCH_SIZE, shuffle=False)
test_loader       = DataLoader(KannadaDataset(X_test, augment=False), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. Model Architecture
# ==========================================
class KannadaCNN(nn.Module):
    """
    Compact CNN classifier for 28x28 grayscale Kannada handwritten digits.
    Incorporates 32 and 64 filter convolutional layers, ReLU activations,
    Batch Normalization, Max Pooling, Spatial/Dense Dropout, and Softmax classification.
    """
    def __init__(self, dropout_rate=0.25):
        super(KannadaCNN, self).__init__()
        # Block 1: 28x28 -> 14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        # Block 2: 14x14 -> 7x7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        # Spatial pooling: 7x7 -> 4x4
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dense classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

sample_model = KannadaCNN()
param_count = count_parameters(sample_model)
print(f"\nKannadaCNN Architecture Summary:\nTotal Trainable Parameters: {param_count:,}")

# ==========================================
# 4. Training Engine
# ==========================================
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, use_scheduler='plateau', checkpoint_path='outputs/model/model.pth'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if use_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)
    elif use_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    else:
        scheduler = None

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'epoch_times': []
    }

    best_val_acc = 0.0
    patience = 4
    patience_counter = 0
    total_start_time = time.time()

    print(f"\n--- Starting Training for {epochs} Epochs ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['lr'].append(float(current_lr))
        history['epoch_times'].append(float(epoch_time))

        if use_scheduler == 'plateau':
            scheduler.step(val_acc)
        elif use_scheduler == 'cosine':
            scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc*100:.2f}%")
                break

    total_training_time = time.time() - total_start_time
    history['total_training_time'] = float(total_training_time)
    print(f"Training completed in {total_training_time:.1f}s ({total_training_time/60:.2f} mins). Best Val Acc: {best_val_acc*100:.2f}%")
    
    # Reload best weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model, history

# ==========================================
# 5. Model Evaluation Engine
# ==========================================
def evaluate_model(model, data_loader, dataset_name="Validation"):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    prec, rec, f1, sup = precision_recall_fscore_support(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)

    per_class_metrics = {}
    for i in range(10):
        per_class_metrics[str(i)] = {
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'f1_score': float(f1[i]),
            'support': int(sup[i])
        }

    results = {
        'dataset_name': dataset_name,
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }
    return results

# ==========================================
# 6. Execute Baseline Model Training
# ==========================================
print("\n" + "="*50)
print("TRAINING MODEL 1: BASELINE CNN (No Augmentation)")
print("="*50)
baseline_model = KannadaCNN()
baseline_checkpoint = 'outputs/model/baseline_cnn_best.pth'
baseline_model, baseline_history = train_model(
    baseline_model, 
    train_loader_base, 
    val_loader, 
    epochs=10, 
    lr=1e-3, 
    use_scheduler='plateau', 
    checkpoint_path=baseline_checkpoint
)

# Evaluate Baseline on Held-out Validation
baseline_val_eval = evaluate_model(baseline_model, val_loader, dataset_name="Validation (Held-out)")
print(f"\nBaseline Validation Accuracy: {baseline_val_eval['accuracy']*100:.2f}% | Macro F1: {baseline_val_eval['macro_f1']:.4f}")

# Evaluate Baseline on Dig-MNIST (Domain Shift)
baseline_dig_eval = evaluate_model(baseline_model, dig_loader, dataset_name="Dig-MNIST (Domain Shift)")
print(f"Baseline Dig-MNIST Accuracy:  {baseline_dig_eval['accuracy']*100:.2f}% | Macro F1: {baseline_dig_eval['macro_f1']:.4f}")

# Domain Shift Accuracy Gap
domain_shift_gap = (baseline_val_eval['accuracy'] - baseline_dig_eval['accuracy']) * 100
print(f"Baseline Domain-Shift Accuracy Gap (Val - Dig): {domain_shift_gap:.2f}% points")

# ==========================================
# 7. Execute Improved Model Training (Innovation Experiment)
# ==========================================
print("\n" + "="*50)
print("TRAINING MODEL 2: IMPROVED CNN (Label-Preserving Augmentation + Cosine Annealing)")
print("="*50)
improved_model = KannadaCNN()
improved_checkpoint = 'outputs/model/improved_cnn_best.pth'
improved_model, improved_history = train_model(
    improved_model, 
    train_loader_aug, 
    val_loader, 
    epochs=10, 
    lr=1e-3, 
    use_scheduler='cosine', 
    checkpoint_path=improved_checkpoint
)

# Evaluate Improved on Held-out Validation
improved_val_eval = evaluate_model(improved_model, val_loader, dataset_name="Validation (Held-out)")
print(f"\nImproved Validation Accuracy: {improved_val_eval['accuracy']*100:.2f}% | Macro F1: {improved_val_eval['macro_f1']:.4f}")

# Evaluate Improved on Dig-MNIST (Domain Shift)
improved_dig_eval = evaluate_model(improved_model, dig_loader, dataset_name="Dig-MNIST (Domain Shift)")
print(f"Improved Dig-MNIST Accuracy:  {improved_dig_eval['accuracy']*100:.2f}% | Macro F1: {improved_dig_eval['macro_f1']:.4f}")

improved_domain_shift_gap = (improved_val_eval['accuracy'] - improved_dig_eval['accuracy']) * 100
print(f"Improved Domain-Shift Accuracy Gap (Val - Dig): {improved_domain_shift_gap:.2f}% points")

# ==========================================
# 8. Visualizations and Error Analysis
# ==========================================
print("\n--- Generating Plots and Visualizations ---")

# A. Learning Curves
def plot_learning_curves(history, title_prefix, save_path):
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Loss
    ax1.plot(epochs_range, history['train_loss'], 'o-', label='Training Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], 's--', label='Validation Loss', color='#d62728', linewidth=2)
    ax1.set_title(f'{title_prefix} - Cross-Entropy Loss vs. Epochs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Accuracy
    ax2.plot(epochs_range, [a * 100 for a in history['train_acc']], 'o-', label='Training Accuracy', color='#2ca02c', linewidth=2)
    ax2.plot(epochs_range, [a * 100 for a in history['val_acc']], 's--', label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    ax2.set_title(f'{title_prefix} - Accuracy (%) vs. Epochs', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

plot_learning_curves(baseline_history, "Baseline CNN", "outputs/plots/baseline_learning_curves.png")
plot_learning_curves(improved_history, "Improved CNN (Augmented)", "outputs/plots/improved_learning_curves.png")

# B. Confusion Matrices
def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[f"D{i}" for i in range(10)],
                yticklabels=[f"D{i}" for i in range(10)])
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('Ground Truth Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

plot_confusion_matrix(np.array(baseline_val_eval['confusion_matrix']), "Baseline CNN - Validation Confusion Matrix", "outputs/plots/baseline_val_confusion_matrix.png")
plot_confusion_matrix(np.array(baseline_dig_eval['confusion_matrix']), "Baseline CNN - Dig-MNIST Confusion Matrix (Domain Shift)", "outputs/plots/baseline_dig_confusion_matrix.png")
plot_confusion_matrix(np.array(improved_dig_eval['confusion_matrix']), "Improved CNN - Dig-MNIST Confusion Matrix", "outputs/plots/improved_dig_confusion_matrix.png")

# C. Misclassified Examples on Validation Set (Genuine errors)
kannada_digit_names = {
    0: 'Digit 0 (Sonne)', 1: 'Digit 1 (Ondu)', 2: 'Digit 2 (Eradu)',
    3: 'Digit 3 (Mooru)', 4: 'Digit 4 (Naalku)', 5: 'Digit 5 (Aidu)',
    6: 'Digit 6 (Aaru)', 7: 'Digit 7 (Eylu)', 8: 'Digit 8 (Entu)', 9: 'Digit 9 (Ombattu)'
}

def plot_misclassified_samples(images, true_labels, pred_labels, probs, title, save_path, n_samples=6):
    errors = np.where(true_labels != pred_labels)[0]
    if len(errors) == 0:
        print(f"No errors found for {title}!")
        return []
    
    n_display = min(n_samples, len(errors))
    selected_errors = errors[:n_display]
    
    fig, axes = plt.subplots(1, n_display, figsize=(3 * n_display, 3.5))
    if n_display == 1:
        axes = [axes]
        
    error_details = []
    for idx, err_idx in enumerate(selected_errors):
        ax = axes[idx]
        img = images[err_idx].reshape(28, 28)
        true_l = int(true_labels[err_idx])
        pred_l = int(pred_labels[err_idx])
        conf = float(probs[err_idx][pred_l]) * 100
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {true_l}\nPred: {pred_l} ({conf:.1f}%)", fontsize=10, fontweight='bold', color='darkred')
        ax.axis('off')
        
        error_details.append({
            'sample_index': int(err_idx),
            'true_label': true_l,
            'true_digit_name': kannada_digit_names[true_l],
            'pred_label': pred_l,
            'pred_digit_name': kannada_digit_names[pred_l],
            'confidence': float(conf)
        })
        
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return error_details

val_error_details = plot_misclassified_samples(
    X_val, baseline_val_eval['labels'], baseline_val_eval['preds'], baseline_val_eval['probs'],
    "Baseline CNN - Validation Set Misclassified Samples",
    "outputs/plots/baseline_val_misclassified.png",
    n_samples=6
)

dig_error_details = plot_misclassified_samples(
    X_dig, baseline_dig_eval['labels'], baseline_dig_eval['preds'], baseline_dig_eval['probs'],
    "Baseline CNN - Dig-MNIST Misclassified Samples (Domain Shift)",
    "outputs/plots/baseline_dig_misclassified.png",
    n_samples=6
)

# D. Domain Shift Comparison Chart: Per-Class Accuracy Val vs Dig-MNIST
val_per_class_acc = [baseline_val_eval['per_class'][str(i)]['recall'] * 100 for i in range(10)]
dig_per_class_acc = [baseline_dig_eval['per_class'][str(i)]['recall'] * 100 for i in range(10)]
imp_dig_per_class_acc = [improved_dig_eval['per_class'][str(i)]['recall'] * 100 for i in range(10)]

plt.figure(figsize=(11, 5))
x = np.arange(10)
width = 0.28
plt.bar(x - width, val_per_class_acc, width, label='Validation Accuracy (%)', color='#1f77b4', edgecolor='black')
plt.bar(x, dig_per_class_acc, width, label='Dig-MNIST Baseline Acc (%)', color='#ff7f0e', edgecolor='black')
plt.bar(x + width, imp_dig_per_class_acc, width, label='Dig-MNIST Improved Acc (%)', color='#2ca02c', edgecolor='black')

plt.xlabel('Digit Class (0 - 9)', fontsize=11, fontweight='bold')
plt.ylabel('Accuracy / Recall (%)', fontsize=11, fontweight='bold')
plt.title('Domain Shift Performance Impact: Validation vs. Dig-MNIST per Class', fontsize=13, fontweight='bold')
plt.xticks(x, [f"Class {i}\n({kannada_digit_names[i].split(' ')[1]})" for i in range(10)], fontsize=9)
plt.ylim(50, 105)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig('outputs/plots/domain_shift_per_class_comparison.png', dpi=300)
plt.close()
print("Saved: outputs/plots/domain_shift_per_class_comparison.png")

# ==========================================
# 9. Confusion Analysis (Top Confused Classes)
# ==========================================
cm_val = np.array(baseline_val_eval['confusion_matrix'])
# Zero out diagonal to find off-diagonal maximums
np.fill_diagonal(cm_val, 0)
top_confusions_idx = np.unravel_index(np.argsort(cm_val, axis=None)[::-1], cm_val.shape)

top_confusions = []
for i in range(min(5, len(top_confusions_idx[0]))):
    t_idx = int(top_confusions_idx[0][i])
    p_idx = int(top_confusions_idx[1][i])
    count = int(cm_val[t_idx, p_idx])
    if count > 0:
        top_confusions.append({
            'true_class': t_idx,
            'true_name': kannada_digit_names[t_idx],
            'pred_class': p_idx,
            'pred_name': kannada_digit_names[p_idx],
            'confusion_count': count
        })

print("\nTop Confused Class Pairs on Validation Set:")
for c in top_confusions:
    print(f"True {c['true_class']} ({c['true_name']}) misclassified as {c['pred_class']} ({c['pred_name']}): {c['confusion_count']} times")

# ==========================================
# 10. Generate Final Test Predictions
# ==========================================
print("\n--- Generating Unlabeled Test Predictions (test.csv) ---")
# Use the best overall performing model (Improved CNN or Baseline CNN based on val accuracy)
best_model = improved_model if improved_val_eval['accuracy'] >= baseline_val_eval['accuracy'] else baseline_model
best_model_name = "Improved CNN" if best_model == improved_model else "Baseline CNN"
print(f"Selected best model for test submission: {best_model_name}")

best_model.eval()
test_preds = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = best_model(images)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())

submission_df = pd.DataFrame({
    'id': test_ids,
    'label': test_preds
})

submission_path = 'outputs/submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Generated submission file at: {submission_path}")

# Rigorous schema verification
print("\n--- Verifying Submission Schema ---")
assert list(submission_df.columns) == ['id', 'label'], f"Column mismatch: {submission_df.columns}"
assert len(submission_df) == len(sample_sub), f"Row count mismatch: {len(submission_df)} vs {len(sample_sub)}"
assert submission_df['id'].equals(sample_sub['id']), "ID column mismatch with sample_submission.csv!"
assert submission_df['label'].dtype in [np.int64, np.int32, int], "Predictions must be integer!"
assert submission_df['label'].isnull().sum() == 0, "Submission contains null values!"
assert submission_df['label'].min() >= 0 and submission_df['label'].max() <= 9, "Predictions outside [0, 9]!"
print("Verification PASSED: outputs/submission.csv exactly conforms to sample_submission.csv schema.")
print(f"Class distribution of test predictions:\n{submission_df['label'].value_counts().sort_index().to_dict()}")

# ==========================================
# 11. Save Structured Metrics & History
# ==========================================
metrics_data = {
    'system_environment': {
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'device': str(device),
        'cpu_threads': 4,
        'random_seed': SEED
    },
    'dataset_splits': {
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'dig_mnist_samples': int(len(X_dig)),
        'test_samples': int(len(X_test)),
        'split_ratio': '80% Train / 20% Stratified Validation'
    },
    'model_architecture': {
        'type': 'Compact KannadaCNN (2 Conv Blocks, MaxPool, BatchNorm, Dropout, Dense)',
        'parameter_count': param_count,
        'selected_submission_model': best_model_name
    },
    'baseline_model': {
        'validation': {
            'accuracy': baseline_val_eval['accuracy'],
            'macro_f1': baseline_val_eval['macro_f1'],
            'per_class': baseline_val_eval['per_class'],
            'confusion_matrix': baseline_val_eval['confusion_matrix']
        },
        'dig_mnist': {
            'accuracy': baseline_dig_eval['accuracy'],
            'macro_f1': baseline_dig_eval['macro_f1'],
            'per_class': baseline_dig_eval['per_class'],
            'confusion_matrix': baseline_dig_eval['confusion_matrix']
        },
        'domain_shift_gap_pct': domain_shift_gap,
        'training_time_seconds': baseline_history['total_training_time']
    },
    'improved_model': {
        'validation': {
            'accuracy': improved_val_eval['accuracy'],
            'macro_f1': improved_val_eval['macro_f1'],
            'per_class': improved_val_eval['per_class'],
            'confusion_matrix': improved_val_eval['confusion_matrix']
        },
        'dig_mnist': {
            'accuracy': improved_dig_eval['accuracy'],
            'macro_f1': improved_dig_eval['macro_f1'],
            'per_class': improved_dig_eval['per_class'],
            'confusion_matrix': improved_dig_eval['confusion_matrix']
        },
        'domain_shift_gap_pct': improved_domain_shift_gap,
        'training_time_seconds': improved_history['total_training_time']
    },
    'top_confusions_validation': top_confusions,
    'val_errors_sample': val_error_details,
    'dig_errors_sample': dig_error_details
}

with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=2)
print("\nSaved metrics to outputs/metrics.json")

training_histories = {
    'baseline': baseline_history,
    'improved': improved_history
}
with open('outputs/training_history.json', 'w') as f:
    json.dump(training_histories, f, indent=2)
print("Saved training history to outputs/training_history.json")

print("\n" + "="*50)
print("EXPERIMENT EXECUTION COMPLETE")
print("="*50)
