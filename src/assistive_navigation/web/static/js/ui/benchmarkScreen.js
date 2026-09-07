/**
 * On-device benchmark screen (requirements 4 and 6).
 *
 * The point of this screen is to make model selection an empirical question. It
 * runs each candidate against the real camera on the real device for 20-30
 * seconds and reports the numbers that actually predict whether a walking demo
 * will hold up. Published desktop benchmarks do not.
 *
 * The six required conditions are operator-guided: the screen tells the tester
 * what to do and tags every sample with the active condition, so the results
 * table shows *where* a candidate falls apart, not just an average that hides it.
 * Nothing here can be automated on a phone - "walk normally" is a human action.
 */

import { MODEL_REGISTRY, getModel } from "../config/modelRegistry.js";
import { BACKEND_LABELS, buildBackendPlan } from "../vision/backendSelector.js";
import { FramePipeline } from "../vision/framePipeline.js";
import { InferenceClient } from "../vision/inferenceClient.js";
import { FrameScheduler, DropReason, LatencyWindow } from "../vision/frameScheduler.js";
import { VisionConfig } from "../config/visionConfig.js";

/** The six conditions the brief requires, in run order. */
export const BENCHMARK_CONDITIONS = Object.freeze([
    { id: "stationary", label: "Hold the phone still", seconds: 5 },
    { id: "slow-walk", label: "Walk slowly", seconds: 5 },
    { id: "normal-walk", label: "Walk at normal pace", seconds: 5 },
    { id: "fast-pan", label: "Pan the camera quickly left and right", seconds: 4 },
    { id: "indoor", label: "Point at an indoor scene", seconds: 4 },
    { id: "outdoor", label: "Point at an outdoor scene or a window", seconds: 4 }
]);

/** Columns exactly as specified in the brief. */
const COLUMNS = [
    { key: "model", label: "Model" },
    { key: "backend", label: "Backend" },
    { key: "inputSize", label: "Input" },
    { key: "inferenceFps", label: "Inf FPS" },
    { key: "avgLatency", label: "Avg ms" },
    { key: "p95Latency", label: "P95 ms" },
    { key: "memory", label: "Heap MB" },
    { key: "detections", label: "Det/frame" },
    { key: "dropped", label: "Dropped" },
    { key: "cameraFps", label: "Cam FPS" },
    { key: "verdict", label: "Verdict" }
];

function percentileOf(values, p) {
    if (!values.length) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1)];
}

function heapMb() {
    const mem = typeof performance !== "undefined" ? performance.memory : null;
    return mem && Number.isFinite(mem.usedJSHeapSize) ? Math.round(mem.usedJSHeapSize / 1e6) : null;
}

/**
 * Per-condition and overall accumulator for one candidate.
 */
class CandidateStats {
    constructor() {
        this.latencies = [];
        this.detectionCounts = [];
        this.conditions = new Map();
        this.dropped = 0;
        this.failures = 0;
        this.heapStart = heapMb();
        this.heapPeak = this.heapStart;
        this.startedAtMs = 0;
        this.endedAtMs = 0;
        this.inferenceTimestamps = [];
        this.cameraFrames = 0;
        this.stabilityWindow = new LatencyWindow(90);
    }

    conditionBucket(conditionId) {
        if (!this.conditions.has(conditionId)) {
            this.conditions.set(conditionId, { latencies: [], detections: [], dropped: 0, failures: 0 });
        }
        return this.conditions.get(conditionId);
    }

    recordInference(conditionId, latencyMs, detectionCount, nowMs) {
        this.latencies.push(latencyMs);
        this.detectionCounts.push(detectionCount);
        this.stabilityWindow.push(latencyMs);
        this.inferenceTimestamps.push(nowMs);
        const bucket = this.conditionBucket(conditionId);
        bucket.latencies.push(latencyMs);
        bucket.detections.push(detectionCount);

        const heap = heapMb();
        if (heap !== null && (this.heapPeak === null || heap > this.heapPeak)) this.heapPeak = heap;
    }

    recordDrop(conditionId) {
        this.dropped += 1;
        this.conditionBucket(conditionId).dropped += 1;
    }

    recordFailure(conditionId) {
        this.failures += 1;
        this.conditionBucket(conditionId).failures += 1;
    }

    summary(cameraFps) {
        const n = this.latencies.length;
        const durationS = Math.max(0.001, (this.endedAtMs - this.startedAtMs) / 1000);
        const avg = n ? this.latencies.reduce((a, b) => a + b, 0) / n : 0;
        const p95 = percentileOf(this.latencies, 95);
        const p50 = percentileOf(this.latencies, 50);
        const meanDet = this.detectionCounts.length
            ? this.detectionCounts.reduce((a, b) => a + b, 0) / this.detectionCounts.length
            : 0;

        return {
            samples: n,
            inferenceFps: n / durationS,
            avgLatency: avg,
            p50Latency: p50,
            p95Latency: p95,
            // Spike ratio is the number that separates "fast" from "usable".
            spikeRatio: p50 > 0 ? p95 / p50 : 0,
            detections: meanDet,
            dropped: this.dropped,
            failures: this.failures,
            heapStart: this.heapStart,
            heapPeak: this.heapPeak,
            heapGrowth: this.heapStart !== null && this.heapPeak !== null ? this.heapPeak - this.heapStart : null,
            cameraFps,
            durationS,
            perCondition: Array.from(this.conditions.entries()).map(([id, b]) => ({
                id,
                samples: b.latencies.length,
                avgLatency: b.latencies.length ? b.latencies.reduce((x, y) => x + y, 0) / b.latencies.length : 0,
                p95Latency: percentileOf(b.latencies, 95),
                detections: b.detections.length ? b.detections.reduce((x, y) => x + y, 0) / b.detections.length : 0,
                dropped: b.dropped,
                failures: b.failures
            }))
        };
    }
}

/**
 * Judge a candidate.
 *
 * Deliberately weighted towards *predictability*. A candidate averaging 55 ms
 * with a p95 of 400 ms loses to one averaging 85 ms with a p95 of 110 ms,
 * because the first one visibly stutters while you walk and the second does not.
 */
export function verdictFor(summary) {
    const reasons = [];
    let score = 0;

    if (summary.samples < 20) return { grade: "FAILED", score: 0, reasons: ["insufficient samples"] };
    if (summary.failures > summary.samples * 0.05) {
        return { grade: "FAILED", score: 0, reasons: [`${summary.failures} inference failures`] };
    }

    // Sustained rate.
    if (summary.inferenceFps >= 12) score += 40;
    else if (summary.inferenceFps >= 8) score += 30;
    else if (summary.inferenceFps >= 5) score += 18;
    else reasons.push(`only ${summary.inferenceFps.toFixed(1)} inference FPS`);

    // Latency.
    if (summary.p95Latency <= 90) score += 30;
    else if (summary.p95Latency <= 140) score += 20;
    else if (summary.p95Latency <= 220) score += 8;
    else reasons.push(`p95 latency ${Math.round(summary.p95Latency)} ms`);

    // Predictability.
    if (summary.spikeRatio <= 1.8) score += 20;
    else if (summary.spikeRatio <= 3) score += 10;
    else reasons.push(`latency spikes ${summary.spikeRatio.toFixed(1)}x above median`);

    // Memory behaviour over the run.
    if (summary.heapGrowth === null) score += 5;
    else if (summary.heapGrowth <= 40) score += 10;
    else if (summary.heapGrowth <= 120) score += 4;
    else reasons.push(`heap grew ${summary.heapGrowth} MB`);

    // Did it actually see anything? A fast detector that detects nothing during
    // an indoor walk is not a usable detector.
    if (summary.detections >= 0.4) score += 10;
    else reasons.push(`only ${summary.detections.toFixed(2)} detections per frame`);

    const grade = score >= 85 ? "EXCELLENT" : (score >= 65 ? "GOOD" : (score >= 45 ? "MARGINAL" : "POOR"));
    return { grade, score, reasons };
}

export class BenchmarkRunner {
    /**
     * @param {{video:HTMLVideoElement, capabilities:object, profile:object,
     *          onProgress?:Function, onCondition?:Function, onCandidate?:Function}} opts
     */
    constructor({ video, capabilities, profile, onProgress = () => {}, onCondition = () => {}, onCandidate = () => {} }) {
        this.video = video;
        this.capabilities = capabilities;
        this.profile = profile;
        this.onProgress = onProgress;
        this.onCondition = onCondition;
        this.onCandidate = onCandidate;

        this.running = false;
        this.cancelled = false;
        this.results = [];
    }

    /**
     * @param {{modelKeys:string[], backendId?:string, conditions?:Array}} opts
     * @returns {Promise<Array>} one result per candidate
     */
    async run({ modelKeys, backendId = null, conditions = BENCHMARK_CONDITIONS }) {
        this.running = true;
        this.cancelled = false;
        this.results = [];

        const plan = buildBackendPlan(this.capabilities);

        for (const modelKey of modelKeys) {
            if (this.cancelled) break;
            const model = getModel(modelKey);
            if (!model) continue;

            // Pick the backend to measure this candidate on: the caller's
            // choice, or the highest-preference planned backend the model
            // actually supports.
            const target = backendId
                || plan.find((p) => model.supportedBackends.includes(p.id))?.id
                || model.supportedBackends[0];

            const result = await this._runCandidate(model, target, conditions);
            this.results.push(result);
            this.onCandidate(result);
        }

        this.running = false;
        return this.results;
    }

    async _runCandidate(model, backendId, conditions) {
        const label = `${model.label} on ${BACKEND_LABELS[backendId] || backendId}`;
        this.onProgress({ stage: "loading", candidate: label });

        const pipeline = new FramePipeline({ inputSize: model.inputSize || this.profile.inputSize, poolSize: 3 });
        const client = new InferenceClient({
            capabilities: this.capabilities,
            profile: this.profile,
            framePipeline: pipeline,
            onEvent: (name, detail) => this.onProgress({ stage: name, candidate: label, detail })
        });

        const base = {
            modelKey: model.key,
            modelLabel: model.label,
            candidate: model.candidate,
            backendId,
            backendLabel: BACKEND_LABELS[backendId] || backendId,
            inputSize: pipeline.inputSize,
            quantization: model.quantization
        };

        let init;
        try {
            init = await client.initialize({
                restrictToBackendId: backendId,
                restrictToModelKey: model.key
            });
        } catch (err) {
            client.dispose();
            pipeline.dispose();
            return { ...base, ok: false, error: String(err?.message || err) };
        }

        if (!init.ok) {
            const attempt = init.attempts?.[init.attempts.length - 1];
            client.dispose();
            pipeline.dispose();
            return {
                ...base,
                ok: false,
                error: attempt?.reason || init.reason || "initialisation rejected",
                warmup: attempt?.warmup || null
            };
        }

        const stats = new CandidateStats();
        // Benchmark scheduling deliberately starts at the profile ceiling and is
        // allowed to fall as low as 1 FPS. We want to see what the candidate can
        // sustain when pushed, not what a conservative controller lets it do.
        const scheduler = new FrameScheduler(VisionConfig.scheduler, {
            targetFps: this.profile.maxFps,
            maxFps: this.profile.maxFps,
            minFps: 1
        });

        stats.startedAtMs = performance.now();
        let conditionId = conditions[0]?.id || "default";

        for (const condition of conditions) {
            if (this.cancelled) break;
            conditionId = condition.id;
            this.onCondition({ candidate: label, condition, remainingMs: condition.seconds * 1000 });
            await this._runCondition({ client, pipeline, scheduler, stats, condition, label });
        }

        stats.endedAtMs = performance.now();
        const summary = stats.summary(scheduler.cameraFps);
        const verdict = verdictFor(summary);

        client.dispose();
        pipeline.dispose();

        return {
            ...base,
            ok: true,
            fromCache: init.initInfo?.fromCache ?? null,
            modelBytes: init.initInfo?.modelBytes ?? null,
            initMs: init.initInfo?.initMs ?? null,
            warmup: init.warmup || null,
            wasmConfig: init.initInfo?.wasmConfig || null,
            summary,
            verdict
        };
    }

    _runCondition({ client, pipeline, scheduler, stats, condition, label }) {
        return new Promise((resolve) => {
            const endAt = performance.now() + condition.seconds * 1000;
            let rafId = null;

            const tick = async () => {
                if (this.cancelled) {
                    if (rafId) cancelAnimationFrame(rafId);
                    resolve();
                    return;
                }
                const nowMs = performance.now();
                if (nowMs >= endAt) {
                    resolve();
                    return;
                }

                scheduler.tickCamera(nowMs);
                stats.cameraFrames += 1;

                const gate = scheduler.shouldRun(nowMs, {
                    busy: client.busy,
                    canCapture: pipeline.canCapture
                });

                if (gate.run) {
                    scheduler.markSubmitted(nowMs);
                    const result = await client.detect(this.video, nowMs);
                    const doneMs = performance.now();

                    if (result?.detections) {
                        scheduler.markCompleted(doneMs, result.latencyMs, result.totalMs || result.latencyMs);
                        stats.recordInference(condition.id, result.latencyMs, result.detections.length, doneMs);
                    } else if (result?.error) {
                        scheduler.markFailed();
                        stats.recordFailure(condition.id);
                    } else if (result?.skipped) {
                        scheduler.markDropped(result.skipped);
                        if (result.skipped !== DropReason.RATE_LIMIT) stats.recordDrop(condition.id);
                    }
                    scheduler.control(doneMs);
                } else if (gate.reason && gate.reason !== DropReason.RATE_LIMIT) {
                    scheduler.markDropped(gate.reason);
                    stats.recordDrop(condition.id);
                }

                this.onCondition({
                    candidate: label,
                    condition,
                    remainingMs: Math.max(0, endAt - performance.now()),
                    live: {
                        inferenceFps: Number(scheduler.inferenceFps.toFixed(1)),
                        p95: scheduler.metrics().latencyP95,
                        samples: stats.latencies.length
                    }
                });

                rafId = requestAnimationFrame(tick);
            };

            rafId = requestAnimationFrame(tick);
        });
    }

    cancel() {
        this.cancelled = true;
    }
}

/**
 * DOM wrapper. Builds the candidate picker, condition prompt and results table.
 */
export class BenchmarkScreen {
    /**
     * @param {{container:HTMLElement, video:HTMLElement, getContext:()=>object,
     *          onBeforeRun?:Function, onAfterRun?:Function}} opts
     */
    constructor({ container, video, getContext, onBeforeRun = async () => {}, onAfterRun = async () => {} }) {
        this.container = container;
        this.video = video;
        this.getContext = getContext;
        this.onBeforeRun = onBeforeRun;
        this.onAfterRun = onAfterRun;
        this.runner = null;
        this.visible = false;

        if (container) this._build();
    }

    _build() {
        this.container.innerHTML = "";
        this.container.setAttribute("aria-label", "On-device model benchmark");

        const intro = document.createElement("p");
        intro.className = "bench-intro";
        intro.textContent = "Measures each detector on this device and camera. Follow the on-screen instruction for each phase.";
        this.container.appendChild(intro);

        const picker = document.createElement("div");
        picker.className = "bench-picker";
        this.checkboxes = new Map();

        for (const model of MODEL_REGISTRY) {
            const id = `bench-${model.key}`;
            const row = document.createElement("label");
            row.className = "bench-option";
            row.setAttribute("for", id);

            const box = document.createElement("input");
            box.type = "checkbox";
            box.id = id;
            box.value = model.key;
            // Default to one representative per candidate letter.
            box.checked = ["yolo11n-320-uint8", "yolo26n-320-uint8", "efficientdet-lite0-int8"].includes(model.key);

            const text = document.createElement("span");
            text.textContent = `${model.candidate === "fallback" ? "Fallback" : `Candidate ${model.candidate}`} · ${model.label} (${model.quantization})`;

            row.appendChild(box);
            row.appendChild(text);
            picker.appendChild(row);
            this.checkboxes.set(model.key, box);
        }
        this.container.appendChild(picker);

        const controls = document.createElement("div");
        controls.className = "bench-controls";

        this.startBtn = document.createElement("button");
        this.startBtn.className = "btn-large btn-primary";
        this.startBtn.textContent = "Run benchmark";
        this.startBtn.addEventListener("click", () => this.start());

        this.cancelBtn = document.createElement("button");
        this.cancelBtn.className = "btn-large btn-secondary";
        this.cancelBtn.textContent = "Cancel";
        this.cancelBtn.disabled = true;
        this.cancelBtn.addEventListener("click", () => this.runner?.cancel());

        controls.appendChild(this.startBtn);
        controls.appendChild(this.cancelBtn);
        this.container.appendChild(controls);

        this.prompt = document.createElement("div");
        this.prompt.className = "bench-prompt";
        this.prompt.setAttribute("role", "status");
        this.prompt.setAttribute("aria-live", "polite");
        this.prompt.textContent = "Idle.";
        this.container.appendChild(this.prompt);

        this.table = document.createElement("table");
        this.table.className = "bench-table";
        const thead = document.createElement("thead");
        const headRow = document.createElement("tr");
        for (const col of COLUMNS) {
            const th = document.createElement("th");
            th.textContent = col.label;
            th.scope = "col";
            headRow.appendChild(th);
        }
        thead.appendChild(headRow);
        this.table.appendChild(thead);
        this.tbody = document.createElement("tbody");
        this.table.appendChild(this.tbody);
        this.container.appendChild(this.table);

        this.detail = document.createElement("pre");
        this.detail.className = "bench-detail";
        this.container.appendChild(this.detail);
    }

    setVisible(visible) {
        this.visible = visible;
        if (this.container) this.container.style.display = visible ? "block" : "none";
        return visible;
    }

    selectedModelKeys() {
        const keys = [];
        for (const [key, box] of this.checkboxes) if (box.checked) keys.push(key);
        return keys;
    }

    async start() {
        const keys = this.selectedModelKeys();
        if (keys.length === 0) {
            this.prompt.textContent = "Select at least one candidate.";
            return;
        }

        const ctx = this.getContext();
        if (!ctx?.capabilities || !ctx?.profile) {
            this.prompt.textContent = "Device profile not ready.";
            return;
        }

        this.startBtn.disabled = true;
        this.cancelBtn.disabled = false;
        this.tbody.innerHTML = "";
        this.detail.textContent = "";

        // The live pipeline and the benchmark must not compete for the camera or
        // the CPU, so assistance is suspended for the duration.
        await this.onBeforeRun();

        const totalSeconds = BENCHMARK_CONDITIONS.reduce((s, c) => s + c.seconds, 0);
        this.prompt.textContent = `Starting. ${keys.length} candidate(s) x ${totalSeconds}s.`;

        this.runner = new BenchmarkRunner({
            video: this.video,
            capabilities: ctx.capabilities,
            profile: ctx.profile,
            onProgress: ({ stage, candidate }) => {
                this.prompt.textContent = `${candidate}: ${stage}…`;
            },
            onCondition: ({ candidate, condition, remainingMs, live }) => {
                const secs = Math.ceil(remainingMs / 1000);
                const liveText = live ? ` · ${live.inferenceFps} FPS, p95 ${live.p95} ms, ${live.samples} samples` : "";
                this.prompt.textContent = `${candidate} — ${condition.label} (${secs}s)${liveText}`;
            },
            onCandidate: (result) => this._appendRow(result)
        });

        try {
            const results = await this.runner.run({ modelKeys: keys });
            this.prompt.textContent = this._concludeText(results);
            this.detail.textContent = JSON.stringify(results, null, 2);
        } catch (err) {
            this.prompt.textContent = `Benchmark error: ${err?.message || err}`;
        } finally {
            this.startBtn.disabled = false;
            this.cancelBtn.disabled = true;
            await this.onAfterRun();
        }
    }

    _concludeText(results) {
        const ok = results.filter((r) => r.ok);
        if (ok.length === 0) return "No candidate completed. See the details below.";
        const best = ok.slice().sort((a, b) => b.verdict.score - a.verdict.score)[0];
        return `Best on this device: ${best.modelLabel} on ${best.backendLabel} — ${best.verdict.grade} (${best.verdict.score}/100).`;
    }

    _appendRow(result) {
        const tr = document.createElement("tr");
        const cells = result.ok
            ? {
                model: result.modelLabel,
                backend: result.backendLabel,
                inputSize: `${result.inputSize}`,
                inferenceFps: result.summary.inferenceFps.toFixed(1),
                avgLatency: Math.round(result.summary.avgLatency),
                p95Latency: Math.round(result.summary.p95Latency),
                memory: result.summary.heapPeak === null
                    ? "n/a"
                    : `${result.summary.heapPeak} (+${result.summary.heapGrowth})`,
                detections: result.summary.detections.toFixed(2),
                dropped: result.summary.dropped,
                cameraFps: result.summary.cameraFps.toFixed(1),
                verdict: `${result.verdict.grade} ${result.verdict.score}`
            }
            : {
                model: result.modelLabel,
                backend: result.backendLabel,
                inputSize: `${result.inputSize}`,
                inferenceFps: "-",
                avgLatency: "-",
                p95Latency: "-",
                memory: "-",
                detections: "-",
                dropped: "-",
                cameraFps: "-",
                verdict: `REJECTED: ${result.error}`
            };

        for (const col of COLUMNS) {
            const td = document.createElement("td");
            td.textContent = String(cells[col.key]);
            if (col.key === "verdict") td.className = `bench-verdict ${result.ok ? result.verdict.grade.toLowerCase() : "failed"}`;
            tr.appendChild(td);
        }
        this.tbody.appendChild(tr);
    }
}

export default BenchmarkScreen;
