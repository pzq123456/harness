"""临时探针: 验证 yolo26n 人检测能否覆盖 harness 框 (用于分类数据集裁剪决策)"""
import glob
import os
import random

from ultralytics import YOLO

BASE = r"C:/Users/admin/Desktop/work/harness/dataset/datasetv1"


def boxes_of(lab):
    out = []
    if os.path.exists(lab):
        with open(lab) as fh:
            for line in fh:
                p = line.strip().split()
                if len(p) == 5:
                    out.append((int(p[0]), *map(float, p[1:])))
    return out


def iou(b1, b2):
    ax, ay, aw, ah = b1
    bx, by, bw, bh = b2
    x1, y1 = max(ax - aw / 2, bx - bw / 2), max(ay - ah / 2, by - bh / 2)
    x2, y2 = min(ax + aw / 2, bx + bw / 2), min(ay + ah / 2, by + bh / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def lab_of(im):
    im = im.replace("\\", "/")
    return im.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


random.seed(1)
imgs = glob.glob(BASE + "/train/images/*.jpg")
pos_imgs = []
for im in imgs:
    if any(c == 0 for c, *_ in boxes_of(lab_of(im))):
        pos_imgs.append(im)
print("train 中有 harness 框的图:", len(pos_imgs))

sample = random.sample(pos_imgs, 40)
model = YOLO(r"C:/Users/admin/Desktop/work/harness/yolo26n.pt")
res = model.predict(sample, conf=0.25, classes=[0], verbose=False, device=0)

det_persons, harness_boxes, overlap_pairs = 0, 0, 0
for im, r in zip(sample, res):
    lab = lab_of(im)
    hbs = [b[1:] for b in boxes_of(lab) if b[0] == 0]
    harness_boxes += len(hbs)
    det_persons += len(r.boxes)
    W, H = r.orig_shape[1], r.orig_shape[0]
    for box in r.boxes.data.cpu().tolist():
        x1, y1, x2, y2 = box[:4]
        pb = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
        if any(iou(pb, h) > 0.1 for h in hbs):
            overlap_pairs += 1

print(
    f"harness boxes={harness_boxes}, detected persons={det_persons}, "
    f"detected person overlapping harness={overlap_pairs}"
)
