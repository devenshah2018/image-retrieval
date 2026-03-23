import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from device import torch_device
from src.model import ResNetTransferModel


@dataclass
class Sample:
    label: int
    path: str


class CaltechDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        image = self.transform(image)
        return image, sample.label


def build_samples(train_entries, data_root):
    samples = []
    for label, rel_path in train_entries:
        full_path = rel_path
        if not os.path.exists(full_path):
            full_path = os.path.join(data_root, rel_path)
        if os.path.exists(full_path):
            samples.append(Sample(label=int(label), path=full_path))
    return samples


def stratified_split(samples, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    train_split = []
    val_split = []

    for _, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        if len(items) == 1:
            n_val = 0
        val_split.extend(items[:n_val])
        train_split.extend(items[n_val:])

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    return train_split, val_split


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Quick trainer for workshop7 webapp model")
    parser.add_argument("--data", default="train_val.json", help="Path to train_val.json")
    parser.add_argument("--output", default="weights/model.pth", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (notebook default: 5e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="CrossEntropyLoss label smoothing (0 disables)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=6000,
        help="Cap train samples for speed (0 disables cap)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch_device()
    print(f"Using device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    train_entries = dataset_json["train"]
    categories = dataset_json["categories"]
    num_classes = len(categories)

    data_root = os.path.dirname(os.path.abspath(args.data))
    samples = build_samples(train_entries, data_root)
    if not samples:
        raise RuntimeError("No valid training images found from train_val.json paths")

    train_samples, val_samples = stratified_split(samples, val_ratio=args.val_ratio, seed=args.seed)

    if args.max_train_samples > 0 and len(train_samples) > args.max_train_samples:
        train_samples = random.sample(train_samples, args.max_train_samples)

    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | Classes: {num_classes}")

    # Augmentations aligned with notebooks/transferlearning-resnet-retrieval.ipynb
    train_tf = transforms.Compose(
        [
            transforms.Resize((144, 144)),
            transforms.RandomResizedCrop(128, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.2)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CaltechDataset(train_samples, train_tf)
    val_ds = CaltechDataset(val_samples, eval_tf)

    num_workers = 2 if os.cpu_count() and os.cpu_count() > 2 else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = ResNetTransferModel(num_classes=num_classes, embedding_size=128, pretrained=True).to(device)
    crit_kwargs = {}
    if args.label_smoothing > 0:
        crit_kwargs["label_smoothing"] = args.label_smoothing
    criterion = nn.CrossEntropyLoss(**crit_kwargs)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-6
    )

    stream_progress = os.environ.get("FLASK_TRAIN_STREAM") == "1" and not sys.stdout.isatty()

    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0
        if stream_progress:
            print(f"Epoch {epoch}/{args.epochs} starting", flush=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y).sum().item()
            running_total += x.size(0)
            progress.set_postfix(
                loss=f"{running_loss / max(running_total, 1):.4f}",
                acc=f"{running_correct / max(running_total, 1):.4f}",
            )
            if stream_progress:
                n = progress.n
                total_batches = len(train_loader)
                step = max(1, total_batches // 20)
                if total_batches and (n % step == 0 or n == total_batches):
                    print(
                        f"Epoch {epoch}/{args.epochs} step {n}/{total_batches} "
                        f"loss={running_loss / max(running_total, 1):.4f} "
                        f"acc={running_correct / max(running_total, 1):.4f}",
                        flush=True,
                    )

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "num_classes": num_classes,
        "categories": categories,
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
    }
    torch.save(checkpoint, args.output)
    print(f"Saved checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
