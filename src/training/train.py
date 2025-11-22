import ssl
import certifi
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from multiprocessing import freeze_support
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.efficientnet_b4 import EfficientNetB4
from utils.dataset_loader import get_dataloaders


# ----------------------- METRICS FUNCTION -----------------------
def compute_metrics(all_labels, all_preds):
    acc = (all_labels == all_preds).mean() * 100
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    return acc, precision, recall, f1
# ---------------------------------------------------------------


def main():
    # Fix SSL verification for downloading pretrained weights if needed
    ssl._create_default_https_context = ssl._create_unverified_context
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: ssl_context

    DATA_DIR = r"D:\1BM22CS181\data"
    SAVE_DIR = r"D:\1BM22CS181\experiments\logs3"
    NUM_EPOCHS = 35
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    RESUME = True
    NUM_WORKERS = 32
    ACCUMULATION_STEPS = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"📂 Loading RGB data from: {DATA_DIR}")
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=4
    )

    print("🔧 Initializing EfficientNet-B4...")
    model = EfficientNetB4(num_classes=2, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    if RESUME:
        if os.path.isdir(SAVE_DIR):
            checkpoint_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".pth")]
            if checkpoint_files:
                latest_ckpt = max(
                    checkpoint_files,
                    key=lambda f: int(f.split("_")[-1].split(".")[0])
                )
                checkpoint_path = os.path.join(SAVE_DIR, latest_ckpt)
                print(f"⏪ Resuming training from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
            else:
                print("⚠️ No checkpoint found, starting training from scratch.")
        else:
            print(f"⚠️ Save directory {SAVE_DIR} does not exist, starting from scratch.")

    os.makedirs(SAVE_DIR, exist_ok=True)


    # ----------------------- TRAINING LOOP -----------------------
    for epoch in range(start_epoch, NUM_EPOCHS):

        # ----------------------- TRAIN -----------------------
        model.train()
        running_loss = 0.0
        train_labels = []
        train_preds = []

        print(f"\n➡️ Starting Epoch {epoch+1}/{NUM_EPOCHS} - Training")
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader, 1):

            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if batch_idx % ACCUMULATION_STEPS == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS

            # predictions
            _, predicted = outputs.max(1)
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())

            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                avg_loss = running_loss / batch_idx
                print(f"   Train Batch [{batch_idx}/{len(train_loader)}] - "
                      f"Loss: {avg_loss:.4f}")

        train_loss = running_loss / len(train_loader)

        train_acc, train_precision, train_recall, train_f1 = compute_metrics(
            np.array(train_labels), np.array(train_preds)
        )

        # ----------------------- VALIDATION -----------------------
        model.eval()
        val_loss = 0.0
        val_labels = []
        val_preds = []

        print(f"\n➡️ Starting Epoch {epoch+1}/{NUM_EPOCHS} - Validation")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader, 1):

                images, labels = images.to(device), labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())

                if batch_idx % 10 == 0 or batch_idx == len(val_loader):
                    avg_loss = val_loss / batch_idx
                    print(f"   Val Batch [{batch_idx}/{len(val_loader)}] - "
                          f"Loss: {avg_loss:.4f}")

        val_loss /= len(val_loader)

        val_acc, val_precision, val_recall, val_f1 = compute_metrics(
            np.array(val_labels), np.array(val_preds)
        )

        # ----------------------- SUMMARY -----------------------
        print(f"\n📊 Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"   🟩 Train → Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"   🟦 Val   → Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # ----------------------- SAVE CHECKPOINT -----------------------
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
        }, checkpoint_path)

        print(f"✅ Saved checkpoint: {checkpoint_path}")


    print("\n🎉 Training complete!")


if __name__ == "__main__":
    freeze_support()
    main()
