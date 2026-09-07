# Field Testing and Acceptance

Procedures for validating the browser client on a physical phone, and the
acceptance criteria the upgrade is measured against.

Safety first, and not as a formality:

- **Never blindfold anyone during development testing.** Use debug mode, a sighted
  observer and a controlled indoor space.
- The person holding the phone walks; the observer watches for hazards and calls
  a halt.
- No traffic, no stairs, no kerbs during development testing. Those are exactly
  the situations the system is least reliable in.
- Obstacles should be soft or light enough to walk into without injury.

---

## 1. Prerequisites

- Served over HTTPS or `localhost`. `getUserMedia` requires a secure context.
- Model assets present in `web/static/models/` (see `docs/benchmarking.md`).
- Rear camera. If the status bar warns that a user-facing camera is in use, stop:
  the scene behind the user is not the path ahead, and guidance will be wrong.
- Volume up, silent switch off. On iOS, speech synthesis needs a user gesture
  first — pressing Start supplies it.

## 2. Start-up sequence

Do not press Start and immediately walk. The system gates on five stages and
announces readiness itself:

```
device profile → model → camera → warm-up → health check → "Vision assistance ready."
```

`Vision ready` in the status pill and the spoken confirmation mean all five have
passed. Only then is walking meaningful. The progress bar shows which stage is in
flight and whether the model came from cache.

Expected first-run versus later-run behaviour:

| Run | Model source | Typical time to ready |
| --- | --- | --- |
| First | Downloaded, then cached in IndexedDB | dominated by the download |
| Later | IndexedDB cache | fast; the status shows `cache hit` |

If the second run is not noticeably faster, the cache is not working. Check the
`Model cache` row in the diagnostics panel; private-browsing and storage-denied
contexts lose caching, and that is reported rather than hidden.

## 3. Ten-station test course

Build the course from chairs, boxes, bins and a volunteer. Run the stations in
order with an observer. Demo mode carries this list with pass/fail marking and a
*Copy results* button (*Developer tools → Demo mode*).

| # | Setup | Expected behaviour | Watch for |
| --- | --- | --- | --- |
| 1 | Clear corridor, nothing within 4 m | Silence. Status CLEAR. | Any speech at all is a failure. |
| 2 | Chair centred in the path | Guidance towards the open side, spoken once | Repetition; a direction into something. |
| 3 | Chair to the left, path open | Silence or one caution | An unnecessary "move right". |
| 4 | Chair to the right, path open | Mirror of station 3 | Left/right asymmetry, which means a sign error. |
| 5 | Person standing centred, 2–3 m | "Person ahead", then a direction if a side is clear | Person not distinguished from furniture. |
| 6 | Person walks across the corridor | "Person crossing"; settles once they leave | A warning that outlives the person. |
| 7 | Two obstacles with a gap between | One stable direction towards the wider gap | Oscillation. |
| 8 | Left blocked, centre partly blocked, right clear | Move right, held stably | Any instruction towards the blocked side. |
| 9 | Right blocked, centre partly blocked, left clear | Move left, held stably | Same. |
| 10 | Both sides and centre blocked | "Stop." immediately, interrupting any speech | Delay; a sidestep instead of a stop. |

Then clear station 10 and confirm "Path clear." is spoken **once**, not
repeatedly.

The same ten situations run headlessly as part of `npm test`
(`js/testing/scenarios.js`). If the phone disagrees with the test suite,
something is device-specific and worth investigating rather than shrugging at.

## 4. Left/right verification

This deserves its own explicit check, because getting it wrong is both easy and
dangerous.

1. Enable the debug overlay (*D*).
2. Place a single obstacle clearly on your left.
3. Confirm the overlay box is drawn on the left of the preview.
4. Confirm the spoken guidance, if any, is consistent with an obstacle on the left.
5. Repeat on the right.
6. Repeat both in portrait **and** landscape.

The diagnostics panel's `Coordinates` row shows the video resolution, display
size, object-fit, mirroring and cropped fraction. In portrait with a 4:3 sensor
the cropped fraction is typically over 0.5 — the sides genuinely are not visible,
and objects there are correctly excluded from reasoning.

## 5. Five-minute walking test

Continuous walking with obstacles, indoors, observer present. Diagnostics panel
open.

Record at start, 1, 2, 3, 4 and 5 minutes:

| Metric | Where | Acceptance |
| --- | --- | --- |
| Camera FPS | Timing | ≥ 24, stable |
| Inference FPS | Timing | ≥ 8 on mid-range, ≥ 5 on low-end |
| P95 latency | Timing | Stable; no growth trend |
| Dropped frames | Timing | Growing slowly; `busy` drops are normal |
| Sustained load | Timing | `stable`, or a single clean step down |
| Heap used / growth | System | No monotonic climb (Chromium only) |
| Health | System | `HEALTHY`; silence under the watchdog timeout |
| Active tracks | Perception | Bounded, typically under 8 |
| Spoken / suppressed | System | Suppressed greatly exceeds spoken |

Must be true at the end:

- no progressive memory growth;
- no growing inference queue — there is no queue, and `Dropped frames` by reason
  proves frames are being skipped rather than buffered;
- no permanent inference freeze;
- no UI lock-up: the preview and buttons stay responsive throughout;
- speech remained useful rather than constant.

## 6. Ten-minute stress test

As above, plus deliberately hostile conditions:

- turn corners repeatedly;
- move between bright and dim areas, and point at a window;
- shake and whip-pan the camera;
- cover the lens for 2–3 seconds, then uncover;
- pause and resume twice;
- stop and restart once;
- background the app and return to it;
- introduce several obstacles at once.

Expected responses:

| Event | Expected |
| --- | --- |
| Turning a corner | Tracks decay, corridor resets, no stale warnings |
| Bright → dim | Brief quality dip, no alarm; recovers |
| Lens covered | `Low visibility`; directions withheld; "Please stop." if it persists |
| Lens uncovered | Returns to normal without intervention |
| Whip-pan | Scene event `VIOLENT_MOTION`, track trust revoked |
| Pause / resume | Clean state on resume; no pre-pause obstacles |
| Backgrounded | Auto-pauses (throttled rAF and paused speech make continuing unsafe) |
| Sustained load | Rate steps down; refinement stage disables itself first |

Must be true at the end: the system recovered from every temporary failure
automatically, and never required a page reload.

## 7. Failure injection

Worth doing at least once, because recovery paths that are never exercised do not
work.

| Injection | How | Expected |
| --- | --- | --- |
| Worker crash | DevTools → Application → terminate the worker | `RESTART_WORKER`, "Reconnecting vision", resumes |
| Backend failure | Disable WebGPU by flag and reload | Falls back to WASM SIMD; diagnostics shows the backend |
| Corrupt cache | Overwrite the IndexedDB entry | Size mismatch drops the entry and re-downloads |
| Camera loss | Revoke camera permission mid-session | `Recovering`, then a clear error and Retry |
| Total failure | Block the model URL and clear the cache | "Vision system unavailable. Please stop." plus Retry |

The system must never respond to any of these by continuing silently as though
nothing happened. That was the worst defect in the previous build.

## 8. Device matrix

Fill this in per device. Tier comes from the diagnostics panel.

| Device | Tier | Backend chosen | Model chosen | Inference FPS | P95 ms | Refinement | 5 min | 10 min | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | |

Cover at minimum: one low-end Android, one mid-range Android, one high-end
Android. Add an iPhone where available. The purpose is to confirm graceful
degradation, not to find the fastest device.

## 9. Acceptance criteria

Headless items are covered by `npm test` (224 checks). Items marked
**physical** require a phone and cannot be verified from a workstation.

| # | Criterion | Verified by |
| --- | --- | --- |
| 1 | Camera stays responsive while AI runs | rAF re-scheduled before inference is fired (`app.js`); **physical** confirmation |
| 2 | No inference queue buildup | `pipeline.test.mjs` no-queue tests; `Dropped frames` by reason |
| 3 | Model is cached | `modelCache.js`; `Model cache` row; **physical** second-run check |
| 4 | Model warm-up works | `inferenceClient.initialize()` gate; `Warm-up p50/1st` row |
| 5 | WebGPU fallback works | `system.test.mjs` plan tests; **physical** flag-disable check |
| 6 | WASM fallback works | Same, plus thread gating tests |
| 7 | Worker recovery works | `system.test.mjs` ladder tests; **physical** injection |
| 8 | Detection continues while walking | `simulation.test.mjs` continuity tests; **physical** |
| 9 | Tracking is stable | `tracker.test.mjs` identity and crossing tests |
| 10 | Direction does not oscillate | `decision.test.mjs` hysteresis; `simulation.test.mjs` reversal counts |
| 11 | Multiple obstacles handled | `decision.test.mjs` multi-object test |
| 12 | Free space influences decisions | `risk.test.mjs`, `freeSpace.test.mjs` |
| 13 | Emergency STOP works | `decision.test.mjs` emergency tests |
| 14 | Speech does not spam | `speech.test.mjs`, `simulation.test.mjs` budget tests |
| 15 | Low visibility handled | `pipeline.test.mjs` quality tests; scenario `low-visibility` |
| 16 | Motion blur handled | `pipeline.test.mjs` blur separation and tolerance tests |
| 17 | Camera rotation handled | `coordinateMapper.test.mjs`; **physical** portrait/landscape check |
| 18 | Memory remains stable | `simulation.test.mjs` bounded-state tests; **physical** heap watch |
| 19 | Five-minute walking test | **physical** (§5) |
| 20 | Ten-minute stress test | **physical** (§6) |
| 21 | Low-end graceful degradation | `system.test.mjs` profile/degradation tests; **physical** |
| 22 | Mid-range stable real time | **physical** |
| 23 | High-end higher-performance profile | **physical** |
| 24 | No raw footage uploaded | No upload path exists in the client |
| 25 | Core guidance works without cloud | Cached model + local pipeline; **physical** offline check |
| 26 | UI never freezes | Non-blocking loop; **physical** |
| 27 | Debug metrics available | Diagnostics panel (*I*) |
| 28 | Real physical-phone testing | **physical** |

### Status

Items 2, 9–16, 21 (logic) and 24 are verified by the automated suite. Items
1, 3, 5–8, 17–23 and 25–28 have their machinery implemented and unit-tested but
their **on-device confirmation is outstanding** — they require a physical
smartphone, which cannot be exercised from a development workstation. Do not
report the project as accepted until §5, §6, §7 and §8 have been completed on real
hardware and recorded.

## 10. What this system is not

Worth restating before any demonstration:

- It is a prototype. It misses things.
- It does not measure distance. Bands are approximate and derived from image
  geometry.
- It does not reliably detect stairs, drops, glass, kerbs or overhead hazards.
- It is not a substitute for a white cane, a guide dog, or a human.

The UI says all of this on screen, and no part of it claims otherwise. Keep it
that way.
