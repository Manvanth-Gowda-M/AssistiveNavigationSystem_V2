"""
Phase 14 optimised pipeline benchmark.
Warms up (10 frames) then measures 60 frames.
Reports mean / median / min / max FPS and per-stage latencies.
Run from project root with venv activated.
"""
import sys, time, statistics
sys.path.insert(0, 'src')

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
try:
    import psutil, os
    _proc = psutil.Process(os.getpid())
    _psutil_ok = True
except ImportError:
    _psutil_ok = False

config = load_config()
cam    = CameraCapture(config)
det    = ObjectDetector(config)
filt   = DetectionFilter(config)
track  = ObjectTracker(config)
depth  = DepthEstimator(config)
fus    = DepthFusion(config)
spa    = SpatialReasoner(config)
pri    = NavigationPriorityEngine(config)
tcf    = TemporalConfirmationFilter(config)
am     = AlertManager(config, audio_queue=None)

print("Loading models ...")
det.load()
depth.load()
cam.open()
print(f"depth.num_threads = {depth._num_threads} (0 = all)")
print(f"detection.num_threads = {config['detection']['num_threads']}")
print()

# ── WARM-UP ────────────────────────────────────────────────────────────────
WARMUP, MEASURE = 10, 60
print(f"Warming up ({WARMUP} frames) ...")
for _ in range(WARMUP):
    frame = cam.read()
    if frame is None: continue
    h, w = frame.shape[:2]
    dr = det.detect(frame)
    fr = filt.filter(dr)
    tracked = track.update(fr.accepted, w, h)
    dm = depth.process_frame(frame)
    fused = fus.fuse(tracked, dm, depth.depth_frame_age)
    directed = spa.assign_direction(fused)
    scored = pri.score(directed)
    confirmed = tcf.update(scored)
    am.process(confirmed)

# ── MEASUREMENT ────────────────────────────────────────────────────────────
print(f"Measuring ({MEASURE} frames) ...")
stages   = ['cam','det','filt','track','depth','fus','spa','pri','temp','alert','total']
timings  = {k: [] for k in stages}
frame_times = []
depth_inferences = 0
ram_samples = []
cpu_samples = []

for i in range(MEASURE):
    if _psutil_ok:
        ram_samples.append(_proc.memory_info().rss // 1024 // 1024)
        cpu_samples.append(psutil.cpu_percent(interval=None))

    t0 = time.perf_counter()

    tc = time.perf_counter(); frame = cam.read(); timings['cam'].append((time.perf_counter()-tc)*1000)
    if frame is None: continue
    h, w = frame.shape[:2]

    tc = time.perf_counter(); dr = det.detect(frame); timings['det'].append((time.perf_counter()-tc)*1000)
    tc = time.perf_counter(); fr = filt.filter(dr); timings['filt'].append((time.perf_counter()-tc)*1000)
    tc = time.perf_counter(); tracked = track.update(fr.accepted, w, h); timings['track'].append((time.perf_counter()-tc)*1000)

    prev = depth.inference_count
    tc = time.perf_counter(); dm = depth.process_frame(frame); timings['depth'].append((time.perf_counter()-tc)*1000)
    if depth.inference_count > prev: depth_inferences += 1

    tc = time.perf_counter(); fused = fus.fuse(tracked, dm, depth.depth_frame_age); timings['fus'].append((time.perf_counter()-tc)*1000)
    tc = time.perf_counter(); directed = spa.assign_direction(fused); timings['spa'].append((time.perf_counter()-tc)*1000)
    tc = time.perf_counter(); scored = pri.score(directed); timings['pri'].append((time.perf_counter()-tc)*1000)
    tc = time.perf_counter(); confirmed = tcf.update(scored); timings['temp'].append((time.perf_counter()-tc)*1000)
    tc = time.perf_counter(); res = am.process(confirmed); timings['alert'].append((time.perf_counter()-tc)*1000)

    total_t = (time.perf_counter()-t0)*1000
    timings['total'].append(total_t)
    frame_times.append(total_t)

cam.release()

# ── REPORT ─────────────────────────────────────────────────────────────────
total_v     = timings['total']
total_mean  = statistics.mean(total_v)
total_med   = statistics.median(total_v)
total_min   = min(total_v)
total_max   = max(total_v)

fps_vals    = [1000.0/t for t in total_v if t > 0]
fps_mean    = statistics.mean(fps_vals)
fps_med     = statistics.median(fps_vals)
fps_min     = min(fps_vals)
fps_max     = max(fps_vals)

print()
print("=" * 62)
print("  PHASE 14 OPTIMISED BENCHMARK")
print(f"  After {WARMUP} warmup frames + {MEASURE} measurement frames")
print("=" * 62)
print()
print("  FPS:")
print(f"    Mean   : {fps_mean:.1f}")
print(f"    Median : {fps_med:.1f}")
print(f"    Min    : {fps_min:.1f}")
print(f"    Max    : {fps_max:.1f}")
print()
print(f"  {'Stage':<12} {'Mean(ms)':>10} {'Median':>9} {'Min':>9} {'Max':>9} {'%Total':>7}")
print("  " + "-"*56)
for k in stages[:-1]:
    v = timings[k]
    if v:
        m   = statistics.mean(v)
        med = statistics.median(v)
        mn  = min(v)
        mx  = max(v)
        pct = (m / total_mean) * 100
        print(f"  {k:<12} {m:10.2f} {med:9.2f} {mn:9.2f} {mx:9.2f} {pct:7.1f}%")
print("  " + "-"*56)
print(f"  {'TOTAL':<12} {total_mean:10.2f} {total_med:9.2f} {total_min:9.2f} {total_max:9.2f}")
print()

# Depth-only when it ran
depth_when_ran = [v for v in timings['depth'] if v > 0]
if depth_when_ran:
    print(f"  Depth (when it ran): mean={statistics.mean(depth_when_ran):.1f}ms  "
          f"min={min(depth_when_ran):.1f}ms  max={max(depth_when_ran):.1f}ms")
expected_depth = MEASURE // config['depth']['depth_update_interval']
print(f"  Depth inferences: {depth_inferences} / {MEASURE} frames  "
      f"(expected ~{expected_depth}, interval={config['depth']['depth_update_interval']})")
print()
if _psutil_ok and ram_samples:
    print(f"  System-wide CPU (psutil, NOT process-specific):")
    print(f"    Mean  : {statistics.mean(cpu_samples):.0f}%   "
          f"Max: {max(cpu_samples):.0f}%")
    print(f"  Process RSS memory (process-specific):")
    print(f"    Mean  : {statistics.mean(ram_samples):.0f} MB  "
          f"Max: {max(ram_samples):.0f} MB")
print("=" * 62)
