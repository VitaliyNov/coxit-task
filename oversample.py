from pathlib import Path
import shutil
import itertools

ROOT = Path("data/cabinet_dataset/train")
TARGET_MIN = 300

for cls_dir in ROOT.iterdir():
    if not cls_dir.is_dir():
        continue

    images = list(cls_dir.glob("*.png"))
    n = len(images)

    if n >= TARGET_MIN:
        continue

    print(f"Oversampling {cls_dir.name}: {n} → {TARGET_MIN}")

    cycle = itertools.cycle(images)
    i = 0
    while len(list(cls_dir.glob("*.png"))) < TARGET_MIN:
        src = next(cycle)
        dst = cls_dir / f"{src.stem}_dup{i}{src.suffix}"
        shutil.copy(src, dst)
        i += 1