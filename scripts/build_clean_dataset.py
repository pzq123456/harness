"""构建干净数据集 dataset_clean/：仅含 harness(0) / no_harness(1) 两类，全部人级框。

来源与规则:
- 正样本: v1 类0(Harness) + v2 类0(harness) + v3 类0(Harness)
- 负样本: v2 类1(no_harness) 显式 + v1 person 框推导(与同图 harness 框 IoU<=0.1)
- 丢弃: v1 类1/3/6 (贴带小框, 语义不一致), v3 无标注图, 跨数据集重复图(v3 的 youtube-41)
- 平衡: 负样本按图片级 1:1 采样至正样本数量
- 划分: 8:1:1 (train/val/test), 固定种子
"""
import glob
import os
import random
import shutil

BASE = r"C:/Users/admin/Desktop/work/harness/dataset"
OUT = r"C:/Users/admin/Desktop/work/harness/dataset_clean"
SEED = 42
SPLIT = (0.8, 0.1, 0.1)
IOU_TH = 0.1  # person 框与 harness 框 IoU 超过此值视为"穿安全带"
SKIP_DUP = {  # 重复图: 保留 v1, 丢弃 v3
    "harness.v3i.yolo26/train/images/youtube-41_jpg.rf.587233f9b55b63f853dd8240307535c3.jpg",
}


def iou(b1, b2):
    ax, ay, aw, ah = b1
    bx, by, bw, bh = b2
    x1, y1 = max(ax - aw / 2, bx - bw / 2), max(ay - ah / 2, by - bh / 2)
    x2, y2 = min(ax + aw / 2, bx + bw / 2), min(ay + ah / 2, by + bh / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def load_image_index(dataset):
    """dataset -> {img_rel_path: (abs_img, [(cls, x, y, w, h)])}"""
    imgs = glob.glob(f"{BASE}/{dataset}/**/images/*", recursive=True)
    imgs = [i for i in imgs if i.lower().endswith((".jpg", ".jpeg", ".png"))]
    index = {}
    for img in imgs:
        rel = img.replace("\\", "/").replace(f"{BASE}/", "")
        if rel in SKIP_DUP:
            continue
        lab = img.replace("\\", "/").replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        boxes = []
        if os.path.exists(lab):
            with open(lab) as fh:
                for line in fh:
                    p = line.strip().split()
                    if len(p) == 5:
                        boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        index[rel] = (img, boxes)
    return index


def remap_v1(rel, boxes, out_boxes, stats):
    """v1: 类0 -> harness; person 框推导负样本; 类1/3/6 丢弃"""
    harness = [b for b in boxes if b[0] == 0]
    for _, x, y, w, h in harness:
        out_boxes.append((0, x, y, w, h))
        stats["pos"] += 1
    for c, x, y, w, h in boxes:
        if c == 7:
            ov = max((iou((x, y, w, h), (bx, by, bw, bh)) for _, bx, by, bw, bh in harness), default=0)
            if ov <= IOU_TH:
                out_boxes.append((1, x, y, w, h))
                stats["neg_derived"] += 1


def remap_v2(rel, boxes, out_boxes, stats):
    """v2: 0 -> harness, 1 -> no_harness"""
    for c, x, y, w, h in boxes:
        if c == 0:
            out_boxes.append((0, x, y, w, h))
            stats["pos"] += 1
        elif c == 1:
            out_boxes.append((1, x, y, w, h))
            stats["neg_explicit"] += 1


def remap_v3(rel, boxes, out_boxes, stats):
    """v3: 0 -> harness"""
    for c, x, y, w, h in boxes:
        if c == 0:
            out_boxes.append((0, x, y, w, h))
            stats["pos"] += 1


def main():
    random.seed(SEED)
    stats = {"pos": 0, "neg_explicit": 0, "neg_derived": 0}

    # 1. 收集所有图片及重映射后的标签
    images = {}  # rel -> (abs_img, [(cls, x,y,w,h)])
    for dataset, remap in [
        ("harness.v1i.yolo26", remap_v1),
        ("Harness.v2i.yolo26", remap_v2),
        ("harness.v3i.yolo26", remap_v3),
    ]:
        for rel, (img, boxes) in load_image_index(dataset).items():
            if not boxes:
                continue  # 丢弃无标注图 (v3 的 318 张)
            out_boxes = []
            remap(rel, boxes, out_boxes, stats)
            if out_boxes:
                images[rel] = (img, out_boxes)

    # 2. 正/负图片池
    pos_imgs = [r for r, (_, bs) in images.items() if any(c == 0 for c, *_ in bs)]
    neg_imgs = [r for r, (_, bs) in images.items() if all(c == 1 for c, *_ in bs)]
    print(f"正样本图片: {len(pos_imgs)}, 负样本图片: {len(neg_imgs)}")
    print(f"正样本框: {stats['pos']}, 显式负样本框: {stats['neg_explicit']}, 推导负样本框: {stats['neg_derived']}")

    # 3. 负样本 1:1 采样 (框级: 使负样本框总数约等于正样本框总数)
    random.shuffle(neg_imgs)
    sel, neg_box_count = [], 0
    for r in neg_imgs:
        if neg_box_count >= stats["pos"]:
            break
        sel.append(r)
        neg_box_count += sum(1 for c, *_ in images[r][1] if c == 1)
    neg_imgs = sel
    print(f"采样后负样本图片: {len(neg_imgs)}, 负样本框: {neg_box_count} (目标 {stats['pos']})")

    # 4. 划分 8:1:1
    all_imgs = pos_imgs + neg_imgs
    random.shuffle(all_imgs)
    n_train = int(len(all_imgs) * SPLIT[0])
    n_val = int(len(all_imgs) * SPLIT[1])
    splits = {
        "train": all_imgs[:n_train],
        "valid": all_imgs[n_train:n_train + n_val],
        "test": all_imgs[n_train + n_val:],
    }

    # 5. 输出
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    counts = {"train": [0, 0], "valid": [0, 0], "test": [0, 0]}
    for split, rels in splits.items():
        img_dir = f"{OUT}/{split}/images"
        lab_dir = f"{OUT}/{split}/labels"
        os.makedirs(img_dir)
        os.makedirs(lab_dir)
        for i, rel in enumerate(rels):
            abs_img, boxes = images[rel]
            ext = os.path.splitext(abs_img)[1]
            src = rel.split("/")[0][:2]  # 源数据集前缀, 避免重名
            name = f"{src}_{i:05d}{ext}"
            shutil.copy2(abs_img, f"{img_dir}/{name}")
            with open(f"{lab_dir}/{os.path.splitext(name)[0]}.txt", "w") as fh:
                for cls, x, y, w, h in boxes:
                    fh.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    counts[split][cls] += 1
        print(f"{split}: {len(rels)} 张图 (harness {counts[split][0]}, no_harness {counts[split][1]})")

    # 6. data.yaml
    with open(f"{OUT}/data.yaml", "w") as fh:
        fh.write("train: train/images\nval: valid/images\ntest: test/images\n\n")
        fh.write("nc: 2\nnames: ['harness', 'no_harness']\n")
    print(f"\ndone -> {OUT}")
    total_pos = sum(c[0] for c in counts.values())
    total_neg = sum(c[1] for c in counts.values())
    print(f"总计: harness {total_pos}, no_harness {total_neg}")


if __name__ == "__main__":
    main()
