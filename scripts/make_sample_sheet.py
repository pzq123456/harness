"""生成标注抽样接触表 (contact sheet) 用于人工目检。

用法:
  python scripts/make_sample_sheet.py --split train --n 50 --out validation/v1_person_no_harness.png
"""
import argparse
import glob
import os
import random

from PIL import Image, ImageDraw, ImageFont

BASE = r"C:/Users/admin/Desktop/work/harness/dataset"

# 类别 -> (名称, 颜色)
CLASS_COLORS = {
    0: ("Harness", (0, 200, 0)),      # 绿
    3: ("harness", (0, 255, 0)),
    6: ("no-harness", (255, 0, 0)),
    7: ("person", (255, 80, 0)),      # 橙
    1: ("No_Harness", (255, 0, 255)),
    2: ("face", (0, 200, 255)),
    4: ("helmet", (0, 128, 255)),
    5: ("machine", (128, 128, 128)),
}


def load_boxes(label_path):
    """读取 YOLO 标签 -> [(class, x_c, y_c, w, h)] (归一化坐标)"""
    boxes = []
    with open(label_path) as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) == 5:
                boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def find_images(dataset):
    imgs = glob.glob(f"{BASE}/{dataset}/**/images/*.jpg", recursive=True)
    imgs += glob.glob(f"{BASE}/{dataset}/**/images/*.jpeg", recursive=True)
    imgs += glob.glob(f"{BASE}/{dataset}/**/images/*.png", recursive=True)
    return imgs


def pick(dataset, split, must_have, must_not_have, n, seed=42):
    """抽样: 图片需含 must_have 中至少一类, 且不含 must_not_have 中任何类"""
    rng = random.Random(seed)
    imgs = find_images(dataset)
    rng.shuffle(imgs)
    picked = []
    for img in imgs:
        if len(picked) >= n:
            break
        lab = img.replace("\\", "/").replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lab):
            continue
        if not ("/valid/" in lab or f"\\valid\\" in lab or "/test/" in lab or f"\\test\\" in lab
                or "/train/" in lab or f"\\train\\" in lab):
            continue
        if f"/{split}/" not in lab and f"\\{split}\\" not in lab:
            continue
        classes = {c for c, *_ in load_boxes(lab)}
        if must_have and not (classes & must_have):
            continue
        if must_not_have and (classes & must_not_have):
            continue
        picked.append((img, lab))
    return picked


def draw_sheet(items, out_path, thumb_w=288, cols=5, label_style="cls"):
    """items: [(img_path, label_path)] -> 画框接触表"""
    thumbs = []
    for img_path, lab_path in items:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        ratio = thumb_w / w
        img = img.resize((thumb_w, int(h * ratio)))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        for cls, xc, yc, bw, bh in load_boxes(lab_path):
            name, color = CLASS_COLORS.get(cls, ("?", (255, 255, 255)))
            x1 = int((xc - bw / 2) * thumb_w)
            y1 = int((yc - bh / 2) * img.height)
            x2 = int((xc + bw / 2) * thumb_w)
            y2 = int((yc + bh / 2) * img.height)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            if label_style == "cls":
                draw.text((x1, max(0, y1 - 16)), f"{cls}:{name}", fill=color, font=font)
        thumbs.append(img)
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.height for t in thumbs) + 24  # 底部留文件名空间
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), (20, 20, 20))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(t, (c * thumb_w, r * cell_h))
        d = ImageDraw.Draw(sheet)
        fname = os.path.basename(items[i][0])
        d.text((c * thumb_w + 4, r * cell_h + t.height + 2), fname[:40], fill=(255, 255, 255))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)
    print(f"saved: {out_path}  ({len(thumbs)} images)")
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="harness.v1i.yolo26")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--must-have", type=int, nargs="*", default=[])
    ap.add_argument("--must-not-have", type=int, nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = pick(
        args.dataset, args.split,
        must_have=set(args.must_have),
        must_not_have=set(args.must_not_have),
        n=args.n, seed=args.seed,
    )
    if not items:
        print("no matches")
    else:
        draw_sheet(items, args.out)
