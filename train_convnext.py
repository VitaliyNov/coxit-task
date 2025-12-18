import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

import timm
from sklearn.metrics import accuracy_score, classification_report


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(imgsz: int):
    # Safe augmentations for cabinet crops (drawings)
    train_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, eval_tf


def make_weighted_sampler(train_ds: datasets.ImageFolder):
    targets = [y for _, y in train_ds.samples]
    class_counts = np.bincount(targets)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_counts.tolist()


@torch.no_grad()
def eval_loader(model, loader, device, class_names):
    model.eval()
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    return acc, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Dataset root with train/ val/ test/")
    ap.add_argument("--model", type=str, default="convnext_tiny",
                    help="timm model name: convnext_tiny / convnext_small / convnext_base")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="runs_convnext")
    ap.add_argument("--use_weighted_sampler", action="store_true",
                    help="Oversample rare classes during training.")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=8, help="Early stopping patience on val acc.")
    args = ap.parse_args()

    seed_all(args.seed)

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    for p in (train_dir, val_dir, test_dir):
        if not p.exists():
            raise FileNotFoundError(f"Missing split folder: {p}")

    outdir = Path(args.outdir) / f"{args.model}_img{args.imgsz}_seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.imgsz)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)

    if val_ds.class_to_idx != train_ds.class_to_idx or test_ds.class_to_idx != train_ds.class_to_idx:
        raise RuntimeError("Class folder mismatch between splits. Ensure same class folders in train/val/test.")

    class_names = train_ds.classes
    num_classes = len(class_names)

    # Dataloaders
    if args.use_weighted_sampler:
        sampler, class_counts = make_weighted_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True)
        (outdir / "class_counts.json").write_text(json.dumps(
            {"classes": class_names, "counts": class_counts}, indent=2
        ), encoding="utf-8")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device if (torch.cuda.is_available() and args.device != "cpu") else "cpu")

    # Model
    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
    model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing) if args.label_smoothing > 0 else nn.CrossEntropyLoss()

    # Optimizer + scheduler (ConvNeXt often likes slightly lower LR than ResNet)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_path = outdir / "best.pt"
    last_path = outdir / "last.pt"
    patience_left = args.patience

    (outdir / "meta.json").write_text(json.dumps({
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "wd": args.wd,
        "seed": args.seed,
        "classes": class_names,
        "use_weighted_sampler": bool(args.use_weighted_sampler),
        "label_smoothing": args.label_smoothing,
    }, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * x.size(0)
            pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        val_acc, _ = eval_loader(model, val_loader, device, class_names)

        torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, last_path)

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_path)
            patience_left = args.patience
        else:
            patience_left -= 1

        print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f}  val_acc={val_acc:.4f}  best_val_acc={best_val_acc:.4f}  patience_left={patience_left}")

        if patience_left <= 0:
            print("Early stopping.")
            break

    # Final evaluation (best checkpoint) on val + test
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    val_acc, val_report = eval_loader(model, val_loader, device, class_names)
    test_acc, test_report = eval_loader(model, test_loader, device, class_names)

    (outdir / "val_report.txt").write_text(f"Val accuracy: {val_acc:.4f}\n\n{val_report}", encoding="utf-8")
    (outdir / "test_report.txt").write_text(f"Test accuracy: {test_acc:.4f}\n\n{test_report}", encoding="utf-8")

    print("\n✅ Training finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Val accuracy:  {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
