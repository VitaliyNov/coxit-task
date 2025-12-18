import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


TARGET_CATEGORIES_DEFAULT = [
    "lc:bcabo",
    "lc:wcabo",
    "lc:muscabinso",
    "lc:wcabcub",
    "lc:bcabocub",
]


def safe_name(s: str) -> str:
    s = s.replace(":", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s.strip("_")


@dataclass
class Ann:
    bbox_xywh: Tuple[float, float, float, float]
    category_id: int
    ann_id: Optional[int] = None


def load_simple_categories(simple_categories_path: Path) -> Dict[int, dict]:
    with simple_categories_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    by_id: Dict[int, dict] = {}
    for rec in data:
        if "id" in rec:
            by_id[int(rec["id"])] = rec
    return by_id


def parse_page_simple_json(page_simple_json_path: Path) -> List[Ann]:
    with page_simple_json_path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    anns: List[Ann] = []
    for a in d.get("annotations", []):
        bbox = a.get("bbox")
        cid = a.get("category_id")
        if bbox is None or cid is None:
            continue
        ann_id = a.get("id", None)
        anns.append(Ann(tuple(bbox), int(cid), ann_id=int(ann_id) if ann_id is not None else None))
    return anns


def clamp_bbox_xyxy(
    x0: float, y0: float, x1: float, y1: float, w: int, h: int
) -> Optional[Tuple[int, int, int, int]]:
    x0i = max(0, min(int(round(x0)), w - 1))
    y0i = max(0, min(int(round(y0)), h - 1))
    x1i = max(0, min(int(round(x1)), w))
    y1i = max(0, min(int(round(y1)), h))
    if x1i <= x0i or y1i <= y0i:
        return None
    return x0i, y0i, x1i, y1i


def pad_bbox_xywh(x: float, y: float, bw: float, bh: float, pad: float) -> Tuple[float, float, float, float]:
    px = bw * pad
    py = bh * pad
    return x - px, y - py, bw + 2 * px, bh + 2 * py


def ensure_dirs(base_out: Path, splits: Iterable[str], class_names: Iterable[str]) -> None:
    for split in splits:
        for cn in class_names:
            (base_out / split / safe_name(cn)).mkdir(parents=True, exist_ok=True)


def find_page_folders(root: Path) -> List[Path]:
    page_folders: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        if "debug" in Path(dirpath).parts:
            continue
        has_png = any(f.lower().endswith(".png") for f in filenames)
        has_simple = any(f.lower().endswith("_simple.json") for f in filenames)
        if has_png and has_simple:
            page_folders.append(Path(dirpath))
    return page_folders


def pick_page_png_and_simple_json(page_dir: Path) -> Tuple[Path, Path]:
    pngs = sorted([p for p in page_dir.iterdir() if p.suffix.lower() == ".png"])
    simples = sorted([p for p in page_dir.iterdir() if p.name.lower().endswith("_simple.json")])
    if not pngs or not simples:
        raise FileNotFoundError(f"Missing png or *_simple.json in {page_dir}")

    png_by_stem = {p.stem: p for p in pngs}
    for sj in simples:
        base_stem = sj.stem.replace("_simple", "")
        if base_stem in png_by_stem:
            return png_by_stem[base_stem], sj

    return pngs[0], simples[0]


def report_split_counts_from_disk(out_dir: Path, classes: Iterable[str], splits: Iterable[str]) -> None:
    print("\n📊 Split report (saved image counts on disk):")
    header = f"{'Class':25s} | " + " | ".join([f"{s:>6s}" for s in splits]) + " | " + f"{'Total':>6s}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for cls in sorted(classes):
        cls_dir = safe_name(cls)
        nums = []
        total = 0
        for sp in splits:
            d = out_dir / sp / cls_dir
            n = len(list(d.glob("*.png"))) if d.exists() else 0
            nums.append(n)
            total += n
        row = f"{cls:25s} | " + " | ".join([f"{n:6d}" for n in nums]) + f" | {total:6d}"
        print(row)
    print("-" * len(header))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to annotated_pdfs_and_data.")
    ap.add_argument("--simple_categories", type=str, default=None, help="Path to simple_categories.json.")
    ap.add_argument("--out", type=str, required=True, help="Output dataset folder.")
    ap.add_argument("--targets", type=str, nargs="*", default=TARGET_CATEGORIES_DEFAULT, help="Target classes.")
    ap.add_argument("--train_frac", type=float, default=0.6, help="Train fraction (default 0.6).")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Val fraction (default 0.2).")
    ap.add_argument("--test_frac", type=float, default=0.2, help="Test fraction (default 0.2).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad", type=float, default=0.05, help="BBox padding fraction.")
    ap.add_argument("--min_size", type=int, default=20, help="Skip tiny crops.")
    ap.add_argument("--max_per_class", type=int, default=0, help="Optional cap per class total.")
    ap.add_argument("--max_mult", type=float, default=1.2,
                    help="Allow split to exceed target fraction by factor (default 1.2).")
    args = ap.parse_args()

    # Validate fractions
    total_frac = args.train_frac + args.val_frac + args.test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0 (got {total_frac})")
    if args.train_frac <= 0 or args.val_frac <= 0 or args.test_frac <= 0:
        raise ValueError("All split fractions must be > 0.")

    root = Path(args.root)
    out = Path(args.out)
    targets = list(dict.fromkeys(args.targets))  # unique, stable
    targets_set = set(targets)

    random.seed(args.seed)

    # Locate simple_categories.json
    if args.simple_categories:
        simple_categories_path = Path(args.simple_categories)
    else:
        candidate = root / "simple_categories.json"
        candidate2 = root.parent / "simple_categories.json"
        if candidate.exists():
            simple_categories_path = candidate
        elif candidate2.exists():
            simple_categories_path = candidate2
        else:
            raise FileNotFoundError("Could not find simple_categories.json. Provide --simple_categories.")

    cat_by_id = load_simple_categories(simple_categories_path)

    # category_id -> category_code (only in_scope)
    id_to_code: Dict[int, str] = {}
    for cid, rec in cat_by_id.items():
        if rec.get("in_scope", False) is True:
            code = rec.get("category")
            if isinstance(code, str):
                id_to_code[cid] = code

    ensure_dirs(out, ["train", "val", "test"], targets)

    page_folders = find_page_folders(root)
    if not page_folders:
        raise RuntimeError(f"No page folders found under {root}")

    # Precompute per-page target counts and total counts
    page_target_counts: Dict[Path, Dict[str, int]] = {}
    total_per_class: Dict[str, int] = {t: 0 for t in targets}

    for p in page_folders:
        try:
            _, sj = pick_page_png_and_simple_json(p)
            anns = parse_page_simple_json(sj)
        except Exception:
            page_target_counts[p] = {t: 0 for t in targets}
            continue

        d = {t: 0 for t in targets}
        for a in anns:
            c = id_to_code.get(a.category_id)
            if c in targets_set:
                d[c] += 1
                total_per_class[c] += 1
        page_target_counts[p] = d

    # Initial random page split by page count
    pages = page_folders[:]
    random.shuffle(pages)

    n_total = len(pages)
    n_train = max(1, int(round(n_total * args.train_frac)))
    n_val = max(1, int(round(n_total * args.val_frac)))
    n_test = max(1, n_total - n_train - n_val)

    # Fix rounding overflow/underflow conservatively
    while n_train + n_val + n_test > n_total:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    while n_train + n_val + n_test < n_total:
        n_train += 1

    train_pages = set(pages[:n_train])
    val_pages = set(pages[n_train:n_train + n_val])
    test_pages = set(pages[n_train + n_val:])

    if len(train_pages) == 0 or len(val_pages) == 0 or len(test_pages) == 0:
        raise RuntimeError("Split produced an empty split; adjust fractions or dataset size.")

    def split_counts(pages_set: set) -> Dict[str, int]:
        c = {t: 0 for t in targets}
        for p in pages_set:
            for t in targets:
                c[t] += page_target_counts[p][t]
        return c

    val_count = split_counts(val_pages)
    test_count = split_counts(test_pages)

    # Helpers for balancing
    def target_instances(frac: float) -> Dict[str, int]:
        return {t: (0 if total_per_class[t] == 0 else max(1, math.ceil(total_per_class[t] * frac))) for t in targets}

    val_target = target_instances(args.val_frac)
    test_target = target_instances(args.test_frac)

    # Promote pages from train -> (val or test) to reach targets (and ensure class presence)
    def promote_to_split(split_name: str, split_pages: set, split_count: Dict[str, int], split_target: Dict[str, int]):
        max_allowed = {t: int(total_per_class[t] * (args.max_mult * (args.val_frac if split_name == "val" else args.test_frac))) + 1
                       for t in targets}

        def would_overfill(p: Path) -> bool:
            for other in targets:
                if total_per_class[other] == 0:
                    continue
                if split_count[other] + page_target_counts[p][other] > max_allowed[other]:
                    return True
            return False

        for cls in targets:
            tot = total_per_class[cls]
            if tot == 0:
                continue

            # Need enough instances for this class in this split
            while split_count[cls] < split_target[cls]:
                candidates = [p for p in train_pages if page_target_counts[p][cls] > 0]
                if not candidates:
                    break  # impossible (class only exists on non-train pages)

                need = split_target[cls] - split_count[cls]

                def score(p: Path) -> float:
                    # prefer pages that add close to 'need' for cls
                    base = abs(page_target_counts[p][cls] - need)
                    # big penalty if it would overfill this split for any class
                    penalty = 1e6 if would_overfill(p) else 0.0
                    return base + penalty

                best = min(candidates, key=score)
                train_pages.remove(best)
                split_pages.add(best)
                for t in targets:
                    split_count[t] += page_target_counts[best][t]

    promote_to_split("val", val_pages, val_count, val_target)
    promote_to_split("test", test_pages, test_count, test_target)

    # De-overfill pass: move pages back to train if a split is overfilled for some class,
    # while keeping at least 1 instance (or target min) for every class that exists.
    def deoverfill_split(split_name: str, split_pages: set, split_count: Dict[str, int], split_target: Dict[str, int]):
        # Minimum requirement: at least 1 instance if class exists at all, and at most the target (best-effort)
        min_req = {t: (0 if total_per_class[t] == 0 else 1) for t in targets}
        # Allow up to target*max_mult
        max_req = {}
        frac = args.val_frac if split_name == "val" else args.test_frac
        for t in targets:
            if total_per_class[t] == 0:
                max_req[t] = 0
            else:
                max_req[t] = int(total_per_class[t] * (frac * args.max_mult)) + 1

        def can_remove(p: Path) -> bool:
            for t in targets:
                if total_per_class[t] == 0:
                    continue
                if split_count[t] - page_target_counts[p][t] < min_req[t]:
                    return False
            return True

        changed = True
        while changed:
            changed = False
            overfilled = [(t, split_count[t] - max_req[t]) for t in targets
                          if total_per_class[t] > 0 and split_count[t] > max_req[t]]
            if not overfilled:
                break

            overfilled.sort(key=lambda x: x[1], reverse=True)
            worst_cls = overfilled[0][0]

            candidates = [p for p in split_pages if page_target_counts[p][worst_cls] > 0 and can_remove(p)]
            if not candidates:
                break

            def removal_score(p: Path) -> int:
                # remove pages that reduce worst_cls most; secondary reduce other overfilled classes too
                score = page_target_counts[p][worst_cls] * 1000
                for t, _ in overfilled[1:]:
                    score += page_target_counts[p][t]
                return score

            remove_page = max(candidates, key=removal_score)
            split_pages.remove(remove_page)
            train_pages.add(remove_page)
            for t in targets:
                split_count[t] -= page_target_counts[remove_page][t]
            changed = True

    deoverfill_split("val", val_pages, val_count, val_target)
    deoverfill_split("test", test_pages, test_count, test_target)

    # ---------------------------
    # Extract crops to disk
    # ---------------------------
    # We’ll count totals as we save (useful when max_per_class is applied)
    saved_total = {t: 0 for t in targets}
    saved_by_split = {sp: {t: 0 for t in targets} for sp in ["train", "val", "test"]}

    skipped_missing_cat = 0
    skipped_small = 0
    skipped_oob = 0

    def which_split(p: Path) -> str:
        if p in val_pages:
            return "val"
        if p in test_pages:
            return "test"
        return "train"

    for page_dir in page_folders:
        split = which_split(page_dir)

        try:
            png_path, simple_json_path = pick_page_png_and_simple_json(page_dir)
        except Exception as e:
            print(f"[WARN] Missing expected files in {page_dir}: {e}")
            continue

        try:
            img = Image.open(png_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open image {png_path}: {e}")
            continue

        W, H = img.size
        anns = parse_page_simple_json(simple_json_path)

        project = page_dir.parent.name
        page = page_dir.name

        for ann in anns:
            cid = ann.category_id
            code = id_to_code.get(cid)

            if code is None:
                skipped_missing_cat += 1
                continue
            if code not in targets_set:
                continue

            if args.max_per_class > 0 and saved_total[code] >= args.max_per_class:
                continue

            x, y, bw, bh = ann.bbox_xywh
            if args.pad > 0:
                x, y, bw, bh = pad_bbox_xywh(x, y, bw, bh, args.pad)

            if bw < args.min_size or bh < args.min_size:
                skipped_small += 1
                continue

            xyxy = clamp_bbox_xyxy(x, y, x + bw, y + bh, W, H)
            if xyxy is None:
                skipped_oob += 1
                continue

            x0, y0, x1, y1 = xyxy
            crop = img.crop((x0, y0, x1, y1))

            ann_id_part = f"{ann.ann_id}" if ann.ann_id is not None else "na"
            out_name = f"{safe_name(project)}__{safe_name(page)}__ann{ann_id_part}__cid{cid}.png"
            out_path = out / split / safe_name(code) / out_name
            crop.save(out_path)

            saved_total[code] += 1
            saved_by_split[split][code] += 1

    # ---------------------------
    # Print summary
    # ---------------------------
    print("\n✅ Done.")
    print(f"Pages total={len(page_folders)}  train={len(train_pages)}  val={len(val_pages)}  test={len(test_pages)}")
    print(f"Fractions target: train={args.train_frac} val={args.val_frac} test={args.test_frac}  seed={args.seed}")

    print("\nSaved instance counts (train/val/test):")
    for t in sorted(targets):
        tot = saved_total[t]
        print(
            f"  {t:15s} "
            f"train={saved_by_split['train'][t]:4d}  "
            f"val={saved_by_split['val'][t]:4d} (target≈{max(1, math.ceil(total_per_class[t]*args.val_frac)) if total_per_class[t] else 0:3d})  "
            f"test={saved_by_split['test'][t]:4d} (target≈{max(1, math.ceil(total_per_class[t]*args.test_frac)) if total_per_class[t] else 0:3d})  "
            f"total={tot:4d}"
        )

    print("\nSkipped:")
    print(f"  missing category_id mapping / not in_scope: {skipped_missing_cat}")
    print(f"  too small (<min_size): {skipped_small}")
    print(f"  invalid/out-of-bounds bbox: {skipped_oob}")

    report_split_counts_from_disk(out, targets, ["train", "val", "test"])


if __name__ == "__main__":
    main()
