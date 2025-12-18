import argparse
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_by_class(split_dir: Path):
    """Return list of (img_path, true_class_name). Assumes ImageFolder layout."""
    items = []
    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name)
    for cls_dir in class_dirs:
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append((p, cls_dir.name))
    return items, [d.name for d in class_dirs]


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

    # annotate values
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_plot[i, j]
            if normalize:
                txt = f"{val:.2f}"
            else:
                txt = str(int(val))
            ax.text(j, i, txt, ha="center", va="center")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_grid(examples, out_path: Path, title: str, cols=4, thumb=240):
    """
    examples: list of dicts with keys: path, true, pred, conf
    Produces a labeled image grid saved to out_path.
    """
    if not examples:
        return

    rows = int(np.ceil(len(examples) / cols))
    W = cols * thumb
    H = rows * thumb + 40

    grid = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # basic font fallback
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

    grid.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to YOLOv8-cls weights, e.g. runs/.../best.pt")
    ap.add_argument("--data_root", type=str, required=True, help="Dataset root containing train/ val/ test/")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--outdir", type=str, default="inference_results")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str, default="0", help='GPU id "0" or "cpu"')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--examples", type=int, default=16, help="How many example images to export per grid")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_path = Path(args.model)
    weights_name = model_path.stem
    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_outdir = Path(args.outdir) / weights_name
    run_outdir.mkdir(parents=True, exist_ok=True)

    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    # Load model
    model = YOLO(str(model_path))

    # Collect images + labels
    items, class_names = list_images_by_class(split_dir)
    if not items:
        raise RuntimeError(f"No images found under: {split_dir}")

    # YOLO's internal names (idx->name). We assume these match folder names used in training.
    # If your training used different naming, you can map here.
    yolo_names = model.names  # dict {idx: "class_name"}
    idx_to_name = {int(k): v for k, v in yolo_names.items()}
    name_to_idx = {v: k for k, v in idx_to_name.items()}

    # Validate folder classes exist in model names
    missing = [c for c in class_names if c not in name_to_idx]
    if missing:
        raise RuntimeError(
            "Some folder class names are not present in model.names:\n"
            f"{missing}\n"
            "Fix by ensuring training folder names and inference folder names match."
        )

    # Run predictions
    records = []
    for i in range(0, len(items), args.batch):
        batch = items[i : i + args.batch]
        paths = [str(p) for p, _ in batch]
        trues = [t for _, t in batch]

        preds = model.predict(paths, device=args.device, verbose=False)
        for pth, true_name, r in zip(paths, trues, preds):
            probs = r.probs.data.detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_name = idx_to_name[pred_idx]
            conf = float(probs[pred_idx])

            records.append(
                {
                    "path": pth,
                    "true": true_name,
                    "pred": pred_name,
                    "conf": conf,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(run_outdir / f"{weights_name}_predictions.csv", index=False)

    # Metrics
    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()

    acc = accuracy_score(y_true, y_pred)
    with open(run_outdir / f"{weights_name}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Split: {args.split}\n")
        f.write(f"Num images: {len(df)}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_true, y_pred, digits=4))

    # Confusion matrices (fixed label order = folder order)
    labels = class_names
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion(
        cm, labels,
        f"Confusion Matrix ({args.split})",
        run_outdir / f"{weights_name}_confusion_raw.png",
        normalize=False
    )

    plot_confusion(
        cm, labels,
        f"Confusion Matrix Normalized ({args.split})",
        run_outdir / f"{weights_name}_confusion_norm.png",
        normalize=True
    )

    # Example exports
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

    print("✅ Inference complete")
    print(f"Split: {args.split}")
    print(f"Images: {len(df)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved to: {run_outdir}")
    print(f"- {weights_name}_predictions.csv")
    print(f"- {weights_name}_metrics.txt")
    print(f"- {weights_name}_confusion_raw.png / {weights_name}_confusion_norm.png")
    print(f"- {weights_name}_examples_best.png / {weights_name}_examples_lowconf_correct.png / {weights_name}_examples_misclassified.png")


if __name__ == "__main__":
    main()
