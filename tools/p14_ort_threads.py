"""Phase 14: ORT thread count benchmark for depth model."""
import sys, time, statistics
sys.path.insert(0,'src')
import numpy as np
import onnxruntime as ort

model_path = 'models/depth_anything_v2_small.onnx'
dummy = np.random.rand(1, 3, 252, 252).astype('float32')

for intra in [0, 2, 4, 6, 8]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    nm = sess.get_inputs()[0].name
    for _ in range(5): sess.run(None, {nm: dummy})
    times = []
    for _ in range(15):
        t0 = time.perf_counter()
        sess.run(None, {nm: dummy})
        times.append((time.perf_counter()-t0)*1000)
    lbl = 'all' if intra==0 else str(intra)
    print(f'intra_threads={lbl:<4}: avg={statistics.mean(times):.1f}ms  min={min(times):.1f}ms  max={max(times):.1f}ms')
