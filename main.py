import cv2
import threading
from ultralytics import YOLO
from pathlib import Path
import time
import numpy as np

class RTSPStreamer:
    """多线程视频流读取类，确保永远获取最新的一帧"""
    def __init__(self, rtsp_url):
        source = int(rtsp_url) if str(rtsp_url).isdigit() else rtsp_url
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret: 
                self.frame = frame
            else: 
                break
        self.cap.release()

    def read(self): 
        return self.frame
    
    def stop(self):
        self.stopped = True
        self.thread.join()

def main():
    script_dir = Path(__file__).parent
    # VEST分类模型路径 (cls模型)
    model_path = script_dir / "runs" / "vest" / "best.pt"
    
    if not model_path.exists():
        print(f"错误：找不到模型文件 {model_path}")
        return

    # 加载模型：base_model用于检测人(COCO类0)，vest_model用于分类人体是否穿戴背心
    print("加载模型中...")
    base_model = YOLO("yolo26n.pt", task="detect")
    vest_model = YOLO(str(model_path), task="classify")  # 改为classify任务
    
    rtsp_url = "rtsp://118.140.234.166:8554/dahua1000352"
    # rtsp_url = "0"  # 使用摄像头测试
    streamer = RTSPStreamer(rtsp_url)
    time.sleep(1)
    
    frame_count = 0
    start_all = time.time()
    fps_display, fps_timer = 0, time.time()
    
    # 定义类别颜色（BGR格式）
    color_map = {
        'vest': (0, 255, 0),        # 绿色 - 穿戴背心
        'no_vest': (0, 0, 255)      # 红色 - 未穿戴背心
    }
    
    try:
        while True:
            t1 = time.time()
            frame = streamer.read()
            if frame is None: 
                continue
            
            annotated_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # 1. 基础模型首先检测人体 (classes=[0] 代表人)
            person_results = base_model.predict(frame, conf=0.4, classes=[0], verbose=False, device=0)
            boxes = person_results[0].boxes.data.cpu().numpy() if len(person_results) > 0 else []
            
            # 2. 遍历检测到的人体，裁剪并分类是否穿戴背心
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                p_conf = box[4]  # 人员检测置信度
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                person_roi = frame[y1:y2, x1:x2]
                
                # 默认类别为 no_vest
                vest_class = "no_vest"
                vest_conf = 0.0
                
                if person_roi.size > 0:
                    # 使用分类模型对裁剪的人体区域进行分类
                    # 分类模型需要输入为 RGB 格式，保持与训练一致
                    roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    
                    # 执行分类推理
                    classify_results = vest_model.predict(
                        roi_rgb, 
                        conf=0.35,  # 分类置信度阈值
                        imgsz=224,  # 分类模型常用输入尺寸
                        verbose=False, 
                        device=0
                    )
                    
                    # 获取分类结果
                    if len(classify_results) > 0:
                        result = classify_results[0]
                        
                        # 获取最高置信度的类别
                        probs = result.probs  # 概率数组
                        if probs is not None:
                            top1_idx = probs.top1  # 最高概率的索引
                            top1_conf = probs.top1conf.item()  # 最高概率值
                            
                            # 获取类别名称
                            class_names = result.names
                            class_name = class_names[top1_idx] if top1_idx in class_names else ""
                            
                            # 判断是否为 vest 类别
                            if class_name.lower() == 'vest':
                                vest_class = "vest"
                                vest_conf = top1_conf
                            else:
                                vest_class = "no_vest"
                                vest_conf = top1_conf
                
                # 3. 绘制人体框及上方标签
                color = color_map.get(vest_class, (0, 0, 255))
                
                # 显示分类结果和置信度
                if vest_class == "vest":
                    label = f"VEST {vest_conf:.2f} (Person {p_conf:.2f})"
                else:
                    label = f"NO VEST {vest_conf:.2f} (Person {p_conf:.2f})"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 性能指标计算
            inference_ms = (time.time() - t1) * 1000
            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = frame_count / (time.time() - start_all)
                fps_timer = time.time()
            
            # 渲染流信息并展示
            cv2.putText(annotated_frame, f"Inf: {inference_ms:.1f}ms  FPS: {fps_display:.1f}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 添加图例说明
            cv2.putText(annotated_frame, "Vest: Green", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "No Vest: Red", (20, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Vest Classification', annotated_frame)
            
            if frame_count % 30 == 0:
                print(f"帧数: {frame_count} | 延迟: {inference_ms:.1f}ms | FPS: {fps_display:.1f}")
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
                
    except Exception as e:
        print(f"异常: {e}")
    finally:
        streamer.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()