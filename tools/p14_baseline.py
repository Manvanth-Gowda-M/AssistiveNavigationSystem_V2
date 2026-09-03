"""Phase 14 baseline benchmark — run from project root."""
import sys, time, statistics
sys.path.insert(0,'src')

from assistive_navigation.utils.config_loader import load_config
from assistive_navigation.camera.capture import CameraCapture
from assistive_navigation.detection.detector import ObjectDetector
from assistive_navigation.detection.filter import DetectionFilter
from assistive_navigation.tracking.tracker import ObjectTracker
from assistive_navigation.depth.depth_estimator import DepthEstimator
from assistive_navigation.depth.fusion import DepthFusion
from assistive_navigation.navigation.spatial import SpatialReasoner
from assistive_navigation.navigation.priority import NavigationPriorityEngine
from assistive_navigation.navigation.temporal import TemporalConfirmationFilter
from assistive_navigation.alerts.alert_manager import AlertManager

config = load_config()
cam   = CameraCapture(config)
det   = ObjectDetector(config)
filt  = DetectionFilter(config)
track = ObjectTracker(config)
depth = DepthEstimator(config)
fus   = DepthFusion(config)
spa   = SpatialReasoner(config)
pri   = NavigationPriorityEngine(config)
tcf   = TemporalConfirmationFilter(config)
am    = AlertManager(config, audio_queue=None)

print("Loading models ...")
det.load()
depth.load()
cam.open()
print("Benchmark starting (60 frames) ...")

N = 60
stages = ['cam','det','filt','track','depth','fus','spa','pri','temp','alert','total']
timings = {k: [] for k in stages}
depth_inferences = 0

for i in range(N):
    t0 = time.perf_counter()

    tc = time.perf_counter()
    frame = cam.read()
    timings['cam'].append((time.perf_counter()-tc)*1000)
    if frame is None:
        continue
    h,w = frame.shape[:2]

    tc = time.perf_counter()
    dr = det.detect(frame)
    timings['det'].append((time.perf_counter()-tc)*1000)

    tc = time.perf_counter()
    fr = filt.filter(dr)
    timings['filt'].append((time.perf_counter()-tc)*1000)

    tc = time.perf_counter()
    tracked = track.update(fr.accepted, w, h)
    timings['track'].append((time.perf_counter()-tc)*1000)

    prev = depth.inference_count
    tc = time.perf_counter()
    dm = depth.process_frame(frame)
    timings['depth'].append((time.perf_counter()-tc)*1000)
    if depth.inference_count > prev:
        depth_inferences += 1

    tc = time.perf_counter()
    fused = fus.fuse(tracked, dm, depth.depth_frame_age)
    timings['fus'].append((time.perf_counter()-tc)*1000)

    tc = time.perf_counter()
    directed = spa.assign_direction(fused)
    timings['spa'].append((time.perf_counter()-tc)*1000)

    tc = time.perf_counter()
    scored = pri.score(directed)
    timings['pri'].append((time.perf_counter()-tc)*1000)

    tc = time.perf_counter()
    confirmed = tcf.update(scored)
    timings['temp'].append((time.perf_counter()-tc)*1000)

    tc = time.perf_counter()
    res = am.process(confirmed)
    timings['alert'].append((time.perf_counter()-tc)*1000)

    timings['total'].append((time.perf_counter()-t0)*1000)

cam.release()

print()
print("=== PER-STAGE TIMING (60 frames, i5-12450H CPU-only) ===")
print(f"{'Stage':<12} {'Mean(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'%Total':>8}")
total_mean = statistics.mean(timings['total'])
print("-"*52)
for k in stages[:-1]:
    v = timings[k]
    if v:
        m = statistics.mean(v)
        mn = min(v)
        mx = max(v)
        pct = (m / total_mean) * 100
        print(f"{k:<12} {m:10.2f} {mn:10.2f} {mx:10.2f} {pct:8.1f}%")

print("-"*52)
v = timings['total']
print(f"{'TOTAL':<12} {statistics.mean(v):10.2f} {min(v):10.2f} {max(v):10.2f}")
print()
fps = 1000.0 / total_mean
print(f"Estimated FPS (1000/mean_total)  : {fps:.1f}")
expected_depth = N // config['depth']['depth_update_interval']
print(f"Depth interval                   : every {config['depth']['depth_update_interval']} frames")
print(f"Depth inferences in {N} frames   : {depth_inferences} (expected ~{expected_depth})")
