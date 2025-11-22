import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ✅ Add parent directory to path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import numpy as np

# --- dataset that loads .npz only ---
from model.dataset_npz_only import FusedNPZDataset


# ------------------ CONFIG ------------------
BASE_DIR = r"D:\1BM22CS181"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "experiments", "multi_fusion_logs6", "best_model.pth")
OUT_DIR = os.path.join(BASE_DIR, "experiments", "cross_eval_results")
os.makedirs(OUT_DIR, exist_ok=True)

# 🔹 Cross-dataset paths
CROSS_TEST_RGB_DIR = r"D:\1BM22CS181\face_forensics\dataset_frames\test"
CROSS_TEST_FREQ_DIR = r"D:\1BM22CS181\face_forensics\dataset_frames_freq\test"

BATCH_SIZE = 16
NUM_WORKERS = 0  # Safe for Windows
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------ TRANSFORMS ------------------
eval_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# ------------------ MODEL DEFINITION ------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=512, num_layers=4, nhead=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=mlp_dim, dropout=dropout,
            activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        if x.ndim == 3:
            return self.encoder(x)
        elif x.ndim == 4:
            B, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            return self.encoder(x)
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")


class NPZFusionModel(nn.Module):
    def __init__(self, token_dim=512, fusion_hidden=768, num_classes=2):
        super().__init__()
        self.token_encoder = TransformerEncoderBlock(embed_dim=token_dim)

        from torchvision import models as tv_models
        self.rgb_cnn = tv_models.efficientnet_b4(weights=tv_models.EfficientNet_B4_Weights.DEFAULT)
        self.freq_cnn = tv_models.efficientnet_b4(weights=tv_models.EfficientNet_B4_Weights.DEFAULT)
        self.rgb_cnn.classifier = nn.Identity()
        self.freq_cnn.classifier = nn.Identity()

        self.cnn_output_dim = 1792
        self.fusion_mlp = nn.Sequential(
            nn.Linear(token_dim + 2 * self.cnn_output_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, features, img_rgb, img_freq):
        token_feat = self.token_encoder(features).mean(dim=1)
        rgb_feat = self.rgb_cnn(img_rgb)
        freq_feat = self.freq_cnn(img_freq)
        fused = torch.cat([token_feat, rgb_feat, freq_feat], dim=1)
        return self.fusion_mlp(fused)


# ------------------ MAIN ------------------
if __name__ == "__main__":
    print("[INFO] Performing CROSS-DATASET testing...")

    # ✅ Load cross-dataset
    test_dataset = FusedNPZDataset(
        BASE_DIR,
        split="test",
        transform=eval_transform,
        rgb_root=CROSS_TEST_RGB_DIR,
        freq_root=CROSS_TEST_FREQ_DIR,
        progress=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"[INFO] Found {len(test_dataset)} test samples in cross dataset.")
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"

    print(f"[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
    model = NPZFusionModel(token_dim=512).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for features, img_rgb, img_freq, labels in test_loader:
            features = features.to(DEVICE, dtype=torch.float32)
            img_rgb = img_rgb.to(DEVICE, dtype=torch.float32)
            img_freq = img_freq.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.long)

            outputs = model(features, img_rgb, img_freq)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ------------------ METRICS ------------------
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\n===== CROSS-DATASET TEST RESULTS =====")
    print(f"Samples : {len(all_labels)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # ------------------ CONFUSION MATRIX ------------------
    cm = confusion_matrix(all_labels, all_preds)
    np.save(os.path.join(OUT_DIR, "confusion_matrix.npy"), cm)
    print(f"[INFO] Confusion matrix saved in {OUT_DIR}")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Real", "Fake"]
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ------------------ ROC Curve ------------------
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    roc_path = os.path.join(OUT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.show()  # 👈 Opens the ROC curve window
    print(f"[INFO] ROC curve saved at: {roc_path}")
    print(f"[INFO] AUC Score: {roc_auc:.4f}")
