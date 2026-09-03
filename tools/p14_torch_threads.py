"""Phase 14: Torch thread count benchmark for YOLO detector."""
import sys, time, statistics, torch
sys.path.insert(0,'src')
import numpy as np
from ultralytics import YOLO

model = YOLO('models/yolo11n.pt', verbose=False)
frame = np.zeros((480, 640, 3), dtype='uint8')

for n in [4, 6, 8, 10, 12]:
    torch.set_num_threads(n)
    # warmup
    for _ in range(5):
        model.predict(frame, conf=0.15, imgsz=640, device='cpu', verbose=False)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict(frame, conf=0.15, imgsz=640, device='cpu', verbose=False)
        times.append((time.perf_counter()-t0)*1000)
    print(f'torch_threads={n:<3}: avg={statistics.mean(times):.1f}ms  min={min(times):.1f}ms  max={max(times):.1f}ms')
