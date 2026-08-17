"""生成分类数据集 datasetv2 的裁剪目检接触表 (contact sheet)。

用法:
  python scripts/make_crop_sheet.py --n 20 --out tmp/datasetv2_sheet.png
"""
import argparse
import glob
import os
import random

from PIL import Image, ImageDraw, ImageFont

BASE = r"C:/Users/admin/Desktop/work/harness/dataset/datasetv2"
CLASSES = ("harness", "no_harness")
CLASS_COLORS = {"harness": (0, 200, 0), "no_harness": (255, 0, 0)}


def pick(cls, split, n, seed=42):
    rng = random.Random(seed)
    files = glob.glob(f"{BASE}/{split}/{cls}/*.jpg")
    rng.shuffle(files)
    return files[:n]


def draw_sheet(items, out_path, thumb_w=200, cols=5):
    thumbs = []
    for path in items:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        ratio = thumb_w / w
        img = img.resize((thumb_w, int(h * ratio)))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        cls = path.replace("\\", "/").split("/")[-2]
        draw.text((4, 4), cls, fill=CLASS_COLORS.get(cls, (255, 255, 255)), font=font)
        thumbs.append(img)
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.height for t in thumbs) + 18
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), (20, 20, 20))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(t, (c * thumb_w, r * cell_h))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)
    print(f"saved: {out_path} ({len(thumbs)} crops)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="每个类别抽几张")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=r"tmp/datasetv2_sheet.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = []
    for cls in CLASSES:
        items += pick(cls, args.split, args.n, seed=args.seed)
    if not items:
        print("no matches")
    else:
        draw_sheet(items, args.out)
