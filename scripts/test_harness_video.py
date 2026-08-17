"""测试脚本：在视频上检测人员并分类 harness(有背心) / no_harness(无背心)。

管线与 main.py 一致: yolo26n 检测人(COCO 类0) -> 裁剪人体 ROI -> 分类模型判 harness/no_harness。
用法:
    python scripts/test_harness_video.py
    python scripts/test_harness_video.py --video "tmp/xxx.mp4" --save
按 q 退出, 结束时打印统计。
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
BASE_MODEL = ROOT / "yolo26n.pt"
CLS_MODEL = ROOT / "runs" / "classify" / "yolo26m_cls_harness_20260817_0547" / "weights" / "best.pt"
DEFAULT_VIDEO = ROOT / "tmp" / "Mobile Camera0676 (1).mp4"
OUT_VIDEO = ROOT / "tmp" / "test_harness_output.mp4"

COLOR_MAP = {
    "harness": (0, 255, 0),     # 绿色 - 有背心
    "no_harness": (0, 0, 255),  # 红色 - 无背心
}


def main():
    parser = argparse.ArgumentParser(description="harness/no_harness 视频测试")
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO), help="视频路径")
    parser.add_argument("--save", action="store_true", help="保存标注输出视频")
    parser.add_argument("--device", type=int, default=0, help="GPU 设备号")
    parser.add_argument("--conf", type=float, default=0.35, help="分类置信度阈值")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"错误：找不到视频文件 {video_path}")
        return
    if not BASE_MODEL.exists():
        print(f"错误：找不到检测模型 {BASE_MODEL}")
        return
    if not CLS_MODEL.exists():
        print(f"错误：找不到分类模型 {CLS_MODEL}")
        return

    print("加载模型中...")
    base_model = YOLO(str(BASE_MODEL), task="detect")
    cls_model = YOLO(str(CLS_MODEL), task="classify")
    print(f"分类类别: {cls_model.names}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    frame_count, person_count = 0, 0
    harness_count, no_harness_count = 0, 0
    start_all = time.time()
    fps_display, fps_timer = 0.0, time.time()
    paused = False

    try:
        while True:
            t1 = time.time()
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
            else:
                ret, frame = True, frame_cache

            if frame is None:
                continue
            frame_cache = frame
            annotated_frame = frame.copy()
            h, w = frame.shape[:2]

            # 1. 检测人体
            person_results = base_model.predict(frame, conf=0.4, classes=[0], verbose=False, device=args.device)
            boxes = person_results[0].boxes.data.cpu().numpy() if len(person_results) > 0 else []

            # 2. 逐个裁剪分类 harness / no_harness
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                p_conf = box[4]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                person_roi = frame[y1:y2, x1:x2]
                cls_name, cls_conf = "no_harness", 0.0

                if person_roi.size > 0:
                    roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    classify_results = cls_model.predict(
                        roi_rgb, conf=args.conf, imgsz=320,
                        verbose=False, device=args.device,
                    )
                    if len(classify_results) > 0 and classify_results[0].probs is not None:
                        probs = classify_results[0].probs
                        top1_idx, top1_conf = probs.top1, probs.top1conf.item()
                        cls_name = classify_results[0].names.get(top1_idx, "no_harness")
                        cls_conf = top1_conf

                person_count += 1
                if cls_name == "harness":
                    harness_count += 1
                else:
                    no_harness_count += 1

                # 3. 绘制
                color = COLOR_MAP.get(cls_name, (0, 0, 255))
                label = f"{cls_name.upper()} {cls_conf:.2f} (Person {p_conf:.2f})"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 性能指标
            inference_ms = (time.time() - t1) * 1000
            if time.time() - fps_timer >= 1.0:
                fps_display = frame_count / (time.time() - start_all)
                fps_timer = time.time()

            cv2.putText(annotated_frame, f"Inf: {inference_ms:.1f}ms  FPS: {fps_display:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Harness: Green", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "No Harness: Red", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if writer is not None:
                writer.write(annotated_frame)

            cv2.imshow("Harness Classification", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

            if frame_count % 30 == 0:
                print(f"帧数: {frame_count} | 延迟: {inference_ms:.1f}ms | FPS: {fps_display:.1f}")

    except Exception as e:
        print(f"异常: {e}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    total = harness_count + no_harness_count
    print(f"\n===== 测试完成 =====")
    print(f"处理帧数: {frame_count} | 总人框数: {person_count} (平均 {person_count / max(frame_count, 1):.2f} 人/帧)")
    print(f"harness: {harness_count} ({harness_count / max(total, 1) * 100:.1f}%) | "
          f"no_harness: {no_harness_count} ({no_harness_count / max(total, 1) * 100:.1f}%)")
    if writer is not None:
        print(f"输出视频: {OUT_VIDEO}")


if __name__ == "__main__":
    main()
