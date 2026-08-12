"""dataset_clean 最终验证: 结构完整性检查 + 可视化抽样接触表。"""
import glob
import os
import random

from PIL import Image, ImageDraw, ImageFont

OUT = r"C:/Users/admin/Desktop/work/harness/dataset_clean"
COLORS = {0: ("harness", (0, 200, 0)), 1: ("no_harness", (255, 40, 40))}


def main():
    # ---- 结构完整性检查 ----
    n_img = n_lab = 0
    for split in ["train", "valid", "test"]:
        imgs = {os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(f"{OUT}/{split}/images/*")}
        labs = {os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(f"{OUT}/{split}/labels/*")}
        assert imgs == labs, f"{split}: 图片与标签不匹配!"
        n_img += len(imgs)
        n_lab += len(labs)
        for p in glob.glob(f"{OUT}/{split}/labels/*"):
            with open(p) as fh:
                lines = [l for l in fh if l.strip()]
            assert lines, f"空标签: {p}"
            for line in lines:
                v = line.split()
                assert len(v) == 5, f"格式错误: {p}: {line}"
                assert 0 <= float(v[1]) <= 1 and 0 <= float(v[2]) <= 1
                assert float(v[3]) <= 1 and float(v[4]) <= 1, f"坐标越界: {p}"
    print(f"结构检查通过: {n_img} 图 = {n_lab} 标签, 无空标签/越界坐标")

    # ---- 可视化抽样 ----
    random.seed(7)
    imgs = glob.glob(f"{OUT}/train/images/*")
    random.shuffle(imgs)
    thumbs = []
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for p in imgs[:30]:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        scale = 320 / w
        img = img.resize((320, max(1, int(h * scale))))
        d = ImageDraw.Draw(img)
        lab = p.replace("\\", "/").replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        with open(lab) as fh:
            for line in fh:
                c, x, y, bw, bh = map(float, line.split())
                name, color = COLORS[int(c)]
                x1, y1 = int((x - bw / 2) * 320), int((y - bh / 2) * img.height)
                x2, y2 = int((x + bw / 2) * 320), int((y + bh / 2) * img.height)
                d.rectangle([x1, y1, x2, y2], outline=color, width=2)
                d.text((x1, max(0, y1 - 14)), name, fill=color, font=font)
        thumbs.append(img)
    cols = 6
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.height for t in thumbs) + 20
    sheet = Image.new("RGB", (cols * 320, rows * cell_h), (20, 20, 20))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(t, (c * 320, r * cell_h))
    out = r"C:/Users/admin/Desktop/work/harness/validation/clean_dataset_sample.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print(f"可视化抽样已保存: {out} ({len(thumbs)} 张)")


if __name__ == "__main__":
    main()
