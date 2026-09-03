"""Phase 14: Detection input_size vs accuracy/speed tradeoff."""
import sys, time, statistics, torch
sys.path.insert(0,'src')
import numpy as np
from ultralytics import YOLO
import cv2

torch.set_num_threads(4)
model = YOLO('models/yolo11n.pt', verbose=False)

# Use a real frame (blank frame is unrealistic for accuracy, but timing is valid)
frame = np.zeros((480, 640, 3), dtype='uint8')

print("Detection input size vs latency (blank frame):")
for sz in [320, 416, 480, 640]:
    for _ in range(5):
        model.predict(frame, conf=0.15, imgsz=sz, device='cpu', verbose=False)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict(frame, conf=0.15, imgsz=sz, device='cpu', verbose=False)
        times.append((time.perf_counter()-t0)*1000)
    print(f'  imgsz={sz:<4}: avg={statistics.mean(times):.1f}ms  min={min(times):.1f}ms')
