import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_eval_transform(imgsz: int):
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def plot_confusion(cm, labels, title, out_path: Path, normalize=False):
    if normalize:
        cm_plot = cm.astype(np.float64)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm_plot / row_sums
    else:
        cm_plot = cm

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(cm_plot)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_plot[i, j]
            txt = f"{val:.2f}" if normalize else str(int(val))
            ax.text(j, i, txt, ha="center", va="center")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_grid(examples, out_path: Path, title: str, cols=4, thumb=240):
    if not examples:
        return

    title = "" if (title is None or title is ...) else str(title)

    rows = int(np.ceil(len(examples) / cols))
    W = cols * thumb
    H = rows * thumb + 40

    grid = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_title = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        font_title = ImageFont.load_default()

    draw.text((10, 10), title, fill=(0, 0, 0), font=font_title)

    y0 = 40
    for idx, ex in enumerate(examples):
        r = idx // cols
        c = idx % cols
        x = c * thumb
        y = y0 + r * thumb

        try:
            img = Image.open(ex["path"]).convert("RGB")
        except Exception:
            continue

        img.thumbnail((thumb, thumb))
        tile = Image.new("RGB", (thumb, thumb), (240, 240, 240))
        tile.paste(img, ((thumb - img.width) // 2, (thumb - img.height) // 2))

        d = ImageDraw.Draw(tile)
        txt = f"T:{ex['true']}\nP:{ex['pred']} ({ex['conf']:.2f})"
        d.rectangle([0, thumb - 48, thumb, thumb], fill=(255, 255, 255))
        d.text((6, thumb - 46), txt, fill=(0, 0, 0), font=font)

        grid.paste(tile, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to EfficientNet checkpoint saved by train script (best.pt / last.pt).")
    ap.add_argument("--model", type=str, default="tf_efficientnet_b0",
                    help="timm model name used for training (must match checkpoint).")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Dataset root containing train/ val/ test/ folders.")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="inference_results_effnet")
    ap.add_argument("--examples", type=int, default=16)
    args = ap.parse_args()

    seed_all(args.seed)

    ckpt_path = Path(args.ckpt)
    weights_name = ckpt_path.stem

    base_outdir = Path(args.outdir)
    run_outdir = base_outdir / weights_name
    run_outdir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    device = torch.device(args.device if (torch.cuda.is_available() and args.device != "cpu") else "cpu")

    # Dataset (ImageFolder)
    tfm = build_eval_transform(args.imgsz)
    ds = datasets.ImageFolder(str(split_dir), transform=tfm)
    class_names = ds.classes
    num_classes = len(class_names)

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    # Model
    model = timm.create_model(args.model, pretrained=False, num_classes=num_classes)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        # assume raw state_dict
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # Inference
    records = []
    idx_to_class = {i: name for i, name in enumerate(class_names)}

    offset = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        pred_idx = probs.argmax(axis=1)
        conf = probs[np.arange(len(pred_idx)), pred_idx]

        y_np = y.numpy()
        batch_paths = [ds.samples[offset + i][0] for i in range(len(y_np))]
        offset += len(y_np)

        for pth, yi, pi, ci in zip(batch_paths, y_np, pred_idx, conf):
            records.append({
                "path": str(pth),
                "true": idx_to_class[int(yi)],
                "pred": idx_to_class[int(pi)],
                "conf": float(ci),
            })

    df = pd.DataFrame(records)
    df.to_csv(run_outdir / f"{weights_name}_predictions.csv", index=False)

    # Metrics
    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    with open(run_outdir / f"{weights_name}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Split: {args.split}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Images: {len(df)}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)

    # Confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plot_confusion(cm, class_names,
                   f"{weights_name} | Confusion Matrix ({args.split})",
                   run_outdir / f"{weights_name}_confusion_raw.png",
                   normalize=False)
    plot_confusion(cm, class_names,
                   f"{weights_name} | Confusion Matrix Normalized ({args.split})",
                   run_outdir / f"{weights_name}_confusion_norm.png",
                   normalize=True)

    # Example grids
    df_sorted = df.sort_values("conf", ascending=False).reset_index(drop=True)
    correct = df_sorted[df_sorted["true"] == df_sorted["pred"]]
    wrong = df_sorted[df_sorted["true"] != df_sorted["pred"]]

    best = correct.head(args.examples).to_dict("records")
    worst = correct.tail(args.examples).to_dict("records")
    mis = wrong.sample(min(args.examples, len(wrong)), random_state=args.seed).to_dict("records") if len(wrong) else []

    make_grid(best, run_outdir / f"{weights_name}_examples_best.png",
              f"{weights_name} | Best correct predictions ({args.split})")
    make_grid(worst, run_outdir / f"{weights_name}_examples_lowconf_correct.png",
              f"{weights_name} | Lowest-confidence correct ({args.split})")
    make_grid(mis, run_outdir / f"{weights_name}_examples_misclassified.png",
              f"{weights_name} | Misclassifications ({args.split})")

    print("✅ EfficientNet inference complete")
    print(f"Split: {args.split} | Images: {len(df)} | Accuracy: {acc:.4f}")
    print(f"Saved to: {run_outdir}")
    print(f"- {weights_name}_predictions.csv")
    print(f"- {weights_name}_metrics.txt")
    print(f"- {weights_name}_confusion_raw.png / {weights_name}_confusion_norm.png")
    print(f"- {weights_name}_examples_*.png")


if __name__ == "__main__":
    main()
