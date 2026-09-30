#!/usr/bin/env python3
"""
Weave - Category Consistency Classifier Training
Trains a lightweight, high-accuracy fashion category classifier (ResNet50 / ConvNet)
on the fashion category taxonomy. Used by EvaluationService to score concept images
for category consistency (target >= 90% agreement).
"""

import os
import json
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm.auto import tqdm

# Local dataset loader
try:
    from ml.training.dataset_utils import (
        FashionDiffusionDataset,
        FashionClassifierDataset,
        FASHION_CATEGORIES,
        CATEGORY_TO_IDX,
        IDX_TO_CATEGORY,
    )
except ImportError:
    from dataset_utils import (
        FashionDiffusionDataset,
        FashionClassifierDataset,
        FASHION_CATEGORIES,
        CATEGORY_TO_IDX,
        IDX_TO_CATEGORY,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Category Consistency Classifier")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing fashion dataset")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/weave/checkpoints/classifier")
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(num_classes: int) -> nn.Module:
    """Builds a pretrained ResNet50 with custom classification head for fashion categories."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Fine-tune layer4 and classification fc
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Using device: {device} for classifier training")

    # 1. Prepare Datasets
    base_ds = FashionDiffusionDataset(data_dir=args.data_dir, size=512)
    full_ds = FashionClassifierDataset(base_ds, is_train=True)

    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = len(FASHION_CATEGORIES)
    logger.info(f"Loaded {train_size} train samples, {val_size} val samples across {num_classes} categories.")

    # 2. Build Model, Criterion, Optimizer
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # 3. Training Loop
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "fashion_classifier_best.pt")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs} [Train]"):
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

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        logger.info(
            f"Epoch {epoch:02d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.1f}%"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "categories": FASHION_CATEGORIES,
                },
                best_model_path,
            )
            logger.info(f"Saved new best model checkpoint to {best_model_path}")

    # Save metadata mapping for downstream FastAPI service
    metadata_path = os.path.join(args.output_dir, "category_labels.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": FASHION_CATEGORIES,
                "category_to_idx": CATEGORY_TO_IDX,
                "idx_to_category": IDX_TO_CATEGORY,
                "best_val_acc": best_val_acc,
            },
            f,
            indent=2,
        )

    print(f"\n[DONE] Category classifier trained! Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Artifacts saved in {args.output_dir}:")
    print(f"  - Model: {best_model_path}")
    print(f"  - Labels: {metadata_path}")


if __name__ == "__main__":
    main()
