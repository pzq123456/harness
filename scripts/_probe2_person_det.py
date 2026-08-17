"""探针2: 评估 yolo26n 人检测对 harness 框的覆盖率 (conf 0.15), 统计漏检原因, 保存裁剪样例"""
import glob
import os
import random

import cv2
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


def lab_of(im):
    im = im.replace("\\", "/")
    return im.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


def iou(b1, b2):
    ax, ay, aw, ah = b1
    bx, by, bw, bh = b2
    x1, y1 = max(ax - aw / 2, bx - bw / 2), max(ay - ah / 2, by - bh / 2)
    x2, y2 = min(ax + aw / 2, bx + bw / 2), min(ay + ah / 2, by + bh / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


random.seed(1)
imgs = glob.glob(BASE + "/train/images/*.jpg")
pos_imgs = [im for im in imgs if any(c == 0 for c, *_ in boxes_of(lab_of(im)))]
print("train 中有 harness 框的图:", len(pos_imgs))

sample = random.sample(pos_imgs, 40)
model = YOLO(r"C:/Users/admin/Desktop/work/harness/yolo26n.pt")
res = model.predict(sample, conf=0.15, classes=[0], verbose=False, device=0)

det_persons, harness_boxes, overlap_pairs = 0, 0, 0
missed_sizes = []
hit_sizes = []
os.makedirs(r"C:/Users/admin/Desktop/work/harness/tmp/probe_crops", exist_ok=True)
n_saved = 0
for idx, (im, r) in enumerate(zip(sample, res)):
    lab = lab_of(im)
    hbs = [b[1:] for b in boxes_of(lab) if b[0] == 0]
    harness_boxes += len(hbs)
    det_persons += len(r.boxes)
    img = cv2.imread(im)
    H, W = img.shape[:2]
    pb_list = []
    for box in r.boxes.data.cpu().tolist():
        x1, y1, x2, y2 = box[:4]
        pb_list.append(((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    for hb in hbs:
        hit = any(iou(hb, pb) > 0.1 for pb in pb_list)
        (hit_sizes if hit else missed_sizes).append(hb[2] * hb[3])
        if hit:
            overlap_pairs += 1
    # 保存前 3 张有重叠的裁剪样例
    for pb in pb_list:
        if any(iou(hb, pb) > 0.1 for hb in hbs) and n_saved < 3:
            x1 = max(0, int((pb[0] - pb[2] / 2) * W))
            y1 = max(0, int((pb[1] - pb[3] / 2) * H))
            x2 = min(W, int((pb[0] + pb[2] / 2) * W))
            y2 = min(H, int((pb[1] + pb[3] / 2) * H))
            cv2.imwrite(
                rf"C:/Users/admin/Desktop/work/harness/tmp/probe_crops/{idx}_{n_saved}.jpg",
                img[y1:y2, x1:x2],
            )
            n_saved += 1

print(f"harness boxes={harness_boxes}, detected persons={det_persons}, "
      f"person overlapping harness={overlap_pairs} ({overlap_pairs / harness_boxes:.0%})")
print(f"被覆盖 harness 框面积中位数={sorted(hit_sizes)[len(hit_sizes)//2]:.4f}, "
      f"漏检 harness 框面积中位数={sorted(missed_sizes)[len(missed_sizes)//2]:.4f}" if hit_sizes and missed_sizes else "")
print(f"saved {n_saved} crop samples -> tmp/probe_crops/")
