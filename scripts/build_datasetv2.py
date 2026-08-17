"""构建分类数据集 datasetv2/：输入为"人框裁剪图"，输出 harness(有背心) / no_harness(无背心) 两类。

背景:
- datasetv1 的 harness(0) 框标的是背心/躯干区域(非全身), no_harness(1) 框是全身人框
- 目标: 正负样本都裁剪"全身人框", 且与推理管线(main.py: yolo26n 检人 -> 裁全身 -> 分类)分布一致
- 方案: yolo26n(COCO) 对全部图重新检测人框; 人框与 harness 框 IoU>0.1 重叠 -> harness, 否则 -> no_harness
- 补充: datasetv1 中人工标注的 no_harness(1) 框本身即全身人框, 并入人框候选池
- 去重: 检测框与人工框 IoU>0.5 视为同一人, 保留人工框
- 划分: 继承 datasetv1 的 train/valid/test, 无跨集泄漏
- 裁剪: 8% 边距 -> 边界 clamp -> 方形补边(114) -> 320x320 (消除正负尺寸差异, 保持比例不变形)
- 过滤: 裁剪后 min(w,h) < 32px 的远景小人丢弃
"""
import glob
import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

SRC = r"C:/Users/admin/Desktop/work/harness/dataset/datasetv1"
OUT = r"C:/Users/admin/Desktop/work/harness/dataset/datasetv2"
MODEL = r"C:/Users/admin/Desktop/work/harness/yolo26n.pt"
DEVICE = 0
CONF = 0.25
IOU_TH = 0.1        # 人框与 harness 框 IoU 超过此值 -> 正样本
DUP_IOU = 0.5       # 检测框与人工框 IoU 超过此值视为同一人
MARGIN = 0.08       # 框外扩边距 (相对宽高的 8%)
SIZE = 320          # 输出方形尺寸
MIN_PX = 32         # 裁剪后最小边长, 更小的丢弃
PAD_VAL = 114       # 补边颜色 (YOLO 惯例灰)


def iou(b1, b2):
    ax, ay, aw, ah = b1
    bx, by, bw, bh = b2
    x1, y1 = max(ax - aw / 2, bx - bw / 2), max(ay - ah / 2, by - bh / 2)
    x2, y2 = min(ax + aw / 2, bx + bw / 2), min(ay + ah / 2, by + bh / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def load_boxes(lab):
    """label -> [(cls, x, y, w, h)] 归一化坐标"""
    out = []
    if os.path.exists(lab):
        with open(lab) as fh:
            for line in fh:
                p = line.strip().split()
                if len(p) == 5:
                    out.append((int(p[0]), *map(float, p[1:])))
    return out


def square_pad_crop(img, x1, y1, x2, y2, size=SIZE, pad_val=PAD_VAL):
    """裁剪 -> 方形补边 -> resize, 返回 BGR 图"""
    crop = img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if max(h, w) > size * 4:  # 超采样: 大图先降采样再补边, 省内存
        scale = size * 4 / max(h, w)
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
        h, w = crop.shape[:2]
    if h == w:
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    side = max(h, w)
    canvas = np.full((side, side, 3), pad_val, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)


def crop_one(img, W, H, b):
    """按归一化框裁剪 + 边距 + 方形补边, 太小返回 None"""
    xc, yc, w, h = b
    mw, mh = w * MARGIN, h * MARGIN
    x1 = max(0, int((xc - w / 2 - mw) * W))
    y1 = max(0, int((yc - h / 2 - mh) * H))
    x2 = min(W, int((xc + w / 2 + mw) * W))
    y2 = min(H, int((yc + h / 2 + mh) * H))
    if x2 - x1 < MIN_PX or y2 - y1 < MIN_PX:
        return None
    return square_pad_crop(img, x1, y1, x2, y2)


def save_one(out, split, cls, im, b, crop, counts):
    """按 原图名_框中心 命名保存, 可追溯"""
    base = os.path.splitext(os.path.basename(im))[0]
    name = f"{base}_{int(b[0] * 1000)}_{int(b[1] * 1000)}.jpg"
    d = f"{out}/{split}/{cls}"
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(f"{d}/{name}", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    counts[split][cls] += 1


def main():
    # 1. 收集每张图: 路径, split, harness(0) 框, 人工 no_harness(1) 人框
    images = []  # (abs_img, split, harness_boxes, manual_person_boxes)
    for split in ("train", "valid", "test"):
        for im in glob.glob(f"{SRC}/{split}/images/*.jpg"):
            lab = im.replace("\\", "/").replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            boxes = load_boxes(lab)
            images.append((im, split,
                           [b[1:] for b in boxes if b[0] == 0],
                           [b[1:] for b in boxes if b[0] == 1]))
    n_harness_img = sum(1 for _, _, h, _ in images if h)
    print(f"图片总数: {len(images)} (含 harness 框的图: {n_harness_img})")

    # 2. yolo26n 全量检测人框 (分批, 避免 batch 过大 OOM)
    model = YOLO(MODEL)
    CHUNK = 256
    all_res = []
    for i in range(0, len(images), CHUNK):
        chunk = [im for im, *_ in images[i:i + CHUNK]]
        all_res += model.predict(
            chunk, conf=CONF, classes=[0], verbose=False, device=DEVICE, batch=32,
        )
        print(f"  检测进度: {min(i + CHUNK, len(images))}/{len(images)}")

    # 3. 打标 + 裁剪
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    counts = {s: {"harness": 0, "no_harness": 0} for s in ("train", "valid", "test")}
    stats = {"det_pos": 0, "det_neg": 0, "man_pos": 0, "man_neg": 0, "dup_drop": 0}

    for (im, split, harness, manual), r in zip(images, all_res):
        img = cv2.imread(im)
        H, W = img.shape[:2]

        # 检测框: 与人工框视为同一人的去重(保留人工框)
        det = []
        for box in r.boxes.data.cpu().tolist():
            x1, y1, x2, y2 = box[:4]
            b = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
            if any(iou(b, m) > DUP_IOU for m in manual):
                stats["dup_drop"] += 1
            else:
                det.append(b)

        # 人框候选池 = 人工框 ∪ 检测框, 统一按 harness 重叠打标
        for b, src in [(m, "man") for m in manual] + [(b, "det") for b in det]:
            is_pos = any(iou(b, h) > IOU_TH for h in harness)
            crop = crop_one(img, W, H, b)
            if crop is None:
                continue
            cls = "harness" if is_pos else "no_harness"
            save_one(OUT, split, cls, im, b, crop, counts)
            stats[f"{src}_{'pos' if is_pos else 'neg'}"] += 1

    # 4. data.yaml (分类格式)
    with open(f"{OUT}/data.yaml", "w") as fh:
        fh.write("train: train\nval: valid\ntest: test\n\n")
        fh.write("nc: 2\nnames: ['harness', 'no_harness']\n")

    # 5. 统计
    print(f"\n人工框->正: {stats['man_pos']}, 人工框->负: {stats['man_neg']}, "
          f"检测框->正: {stats['det_pos']}, 检测框->负: {stats['det_neg']}")
    print(f"检测框与人工框去重丢弃: {stats['dup_drop']}")
    for split, c in counts.items():
        print(f"{split}: harness {c['harness']}, no_harness {c['no_harness']}")
    print(f"\ndone -> {OUT}")


if __name__ == "__main__":
    main()
