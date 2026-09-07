/**
 * Developer performance dashboard (requirement 51).
 *
 * Builds its own DOM so the metric list lives in one place instead of being
 * duplicated across HTML ids and JS lookups. Cells are only written when the
 * value changes, because a 20 Hz dashboard that touches 20 text nodes every tick
 * is a measurable cost on a mid-range phone - and an instrument that distorts
 * the thing it measures is worse than no instrument.
 */

import { cacheStatusSummary } from "../vision/modelCache.js";

/**
 * @typedef {object} MetricSpec
 * @property {string} key
 * @property {string} label
 * @property {string} [group]
 */

/** Metric layout. `key` is looked up in the flattened snapshot. */
const METRICS = [
    { group: "Detector", key: "model", label: "Model" },
    { group: "Detector", key: "backend", label: "Backend" },
    { group: "Detector", key: "inputSize", label: "Input size" },
    { group: "Detector", key: "quantization", label: "Quantisation" },
    { group: "Detector", key: "modelCache", label: "Model cache" },
    { group: "Detector", key: "workerStatus", label: "Worker" },
    { group: "Detector", key: "wasmConfig", label: "WASM config" },
    { group: "Detector", key: "warmup", label: "Warm-up p50/1st" },

    { group: "Timing", key: "cameraFps", label: "Camera FPS" },
    { group: "Timing", key: "inferenceFps", label: "Inference FPS" },
    { group: "Timing", key: "targetFps", label: "Target FPS" },
    { group: "Timing", key: "latencyP50", label: "P50 latency" },
    { group: "Timing", key: "latencyP95", label: "P95 latency" },
    { group: "Timing", key: "endToEndP95", label: "P95 end-to-end" },
    { group: "Timing", key: "droppedFrames", label: "Dropped frames" },
    { group: "Timing", key: "thermal", label: "Sustained load" },

    { group: "Perception", key: "activeTracks", label: "Active tracks" },
    { group: "Perception", key: "confirmedTracks", label: "Confirmed tracks" },
    { group: "Perception", key: "freeSpace", label: "Free space L/C/R" },
    { group: "Perception", key: "passableWidth", label: "Widest gap" },
    { group: "Perception", key: "quality", label: "Frame quality" },
    { group: "Perception", key: "sceneEvent", label: "Scene" },
    { group: "Perception", key: "refinement", label: "Refinement stage" },

    { group: "Decision", key: "pathScores", label: "Path scores L/C/R" },
    { group: "Decision", key: "currentRisk", label: "Current risk" },
    { group: "Decision", key: "decision", label: "Decision" },
    { group: "Decision", key: "decisionConfidence", label: "Confidence" },
    { group: "Decision", key: "switches", label: "Switches / held" },
    { group: "Decision", key: "corridor", label: "Corridor" },

    { group: "System", key: "healthState", label: "Health" },
    { group: "System", key: "degradation", label: "Degradation level" },
    { group: "System", key: "speech", label: "Speech" },
    { group: "System", key: "speechCounters", label: "Spoken / suppressed" },
    { group: "System", key: "recovery", label: "Recovery" },
    { group: "System", key: "memory", label: "Heap used / growth" },
    { group: "System", key: "buffers", label: "Tensor buffers" },
    { group: "System", key: "coords", label: "Coordinates" }
];

export class PerformanceDashboard {
    /**
     * @param {HTMLElement} container
     * @param {{minIntervalMs?:number}} [opts]
     */
    constructor(container, { minIntervalMs = 250 } = {}) {
        this.container = container;
        this.minIntervalMs = minIntervalMs;
        this.cells = new Map();
        this._values = new Map();
        this._lastUpdateMs = 0;
        this.visible = false;
        this._cacheStatus = "checking…";

        if (container) this._build();
        this.refreshCacheStatus();
    }

    _build() {
        this.container.innerHTML = "";
        this.container.setAttribute("role", "region");
        this.container.setAttribute("aria-label", "Developer diagnostics");

        let currentGroup = null;
        let grid = null;

        for (const metric of METRICS) {
            if (metric.group !== currentGroup) {
                currentGroup = metric.group;
                const heading = document.createElement("h3");
                heading.className = "diag-group";
                heading.textContent = currentGroup;
                this.container.appendChild(heading);

                grid = document.createElement("div");
                grid.className = "diag-grid";
                this.container.appendChild(grid);
            }

            const cell = document.createElement("div");
            cell.className = "diag-cell";

            const label = document.createElement("span");
            label.className = "diag-label";
            label.textContent = metric.label;

            const value = document.createElement("span");
            value.className = "diag-value";
            value.textContent = "--";

            cell.appendChild(label);
            cell.appendChild(value);
            grid.appendChild(cell);
            this.cells.set(metric.key, value);
        }
    }

    setVisible(visible) {
        this.visible = visible;
        if (this.container) this.container.style.display = visible ? "block" : "none";
        if (visible) this.refreshCacheStatus();
        return visible;
    }

    async refreshCacheStatus() {
        try {
            const summary = await cacheStatusSummary();
            if (!summary.available) {
                this._cacheStatus = "unavailable";
                return;
            }
            const mb = (summary.totalBytes / 1e6).toFixed(1);
            this._cacheStatus = summary.entries.length
                ? `${summary.entries.length} entr${summary.entries.length === 1 ? "y" : "ies"}, ${mb} MB`
                : "empty";
        } catch {
            this._cacheStatus = "error";
        }
    }

    /**
     * @param {object} s flattened snapshot from the app
     */
    update(s) {
        if (!this.visible || !this.container) return;
        const nowMs = s.nowMs ?? Date.now();
        if (nowMs - this._lastUpdateMs < this.minIntervalMs) return;
        this._lastUpdateMs = nowMs;

        const values = this._flatten(s);
        for (const [key, text] of values) {
            if (this._values.get(key) === text) continue;
            this._values.set(key, text);
            const cell = this.cells.get(key);
            if (cell) cell.textContent = text;
        }
    }

    _flatten(s) {
        const det = s.detector || {};
        const sch = s.scheduler || {};
        const trk = s.tracking || {};
        const fs = s.freeSpace || {};
        const dec = s.decision || {};
        const hp = s.health || {};
        const sp = s.speech || {};
        const rec = s.recovery || {};
        const seg = s.refinement || {};
        const q = s.quality || {};
        const cor = s.corridor || {};
        const scene = s.scene || {};
        const ps = s.pathScores || null;

        const out = new Map();
        const set = (k, v) => out.set(k, v === null || v === undefined ? "--" : String(v));

        set("model", det.modelLabel || "none");
        set("backend", det.backendLabel || "none");
        set("inputSize", det.inputSize ? `${det.inputSize}x${det.inputSize}` : null);
        set("quantization", det.quantization);
        set("modelCache", det.fromCache === null || det.fromCache === undefined
            ? this._cacheStatus
            : `${det.fromCache ? "hit" : "miss"} · ${this._cacheStatus}`);
        set("workerStatus", hp.workerStatus);
        set("wasmConfig", det.wasmConfig
            ? `${det.wasmConfig.numThreads}t simd=${det.wasmConfig.simd ? "on" : "off"}`
            : "n/a");
        set("warmup", det.warmupSteadyStateMs !== null && det.warmupSteadyStateMs !== undefined
            ? `${det.warmupSteadyStateMs} / ${det.warmupFirstPassMs} ms`
            : null);

        set("cameraFps", sch.cameraFps !== undefined ? `${sch.cameraFps}` : null);
        set("inferenceFps", sch.inferenceFps !== undefined ? `${sch.inferenceFps}` : null);
        set("targetFps", sch.targetFps ? `${sch.targetFps} (max ${sch.ceilingFps})` : null);
        set("latencyP50", sch.latencyP50 !== undefined ? `${sch.latencyP50} ms` : null);
        set("latencyP95", sch.latencyP95 !== undefined ? `${sch.latencyP95} ms` : null);
        set("endToEndP95", sch.endToEndP95 !== undefined ? `${sch.endToEndP95} ms` : null);
        set("droppedFrames", sch.droppedFrames !== undefined
            ? `${sch.droppedFrames} (busy ${sch.dropsByReason?.busy || 0})`
            : null);
        set("thermal", sch.thermalThrottled ? `throttled (best ${sch.bestP95} ms)` : "stable");

        set("activeTracks", trk.active);
        set("confirmedTracks", trk.confirmed !== undefined ? `${trk.confirmed} (swaps blocked ${trk.identitySwapGuards})` : null);
        set("freeSpace", fs.left !== undefined ? `${fs.left} / ${fs.center} / ${fs.right}` : null);
        set("passableWidth", fs.widestGapWidth !== undefined
            ? `${(fs.widestGapWidth * 100).toFixed(0)}% ${fs.passable ? "passable" : "TOO NARROW"}`
            : null);
        set("quality", q.state ? `${q.state} ${q.score} (b${q.brightness} blur${q.blur})` : null);
        set("sceneEvent", scene.event ? `${scene.event} Δ${scene.change} shift${scene.shift}` : null);
        set("refinement", seg.enabled === undefined
            ? null
            : (seg.enabled ? `on · ${seg.lastTrigger} · ${seg.p95CostMs} ms` : `off · ${seg.disabledReason || "n/a"}`));

        set("pathScores", ps ? `${ps.left} / ${ps.center} / ${ps.right} → ${ps.best}` : null);
        set("currentRisk", s.maxRisk !== undefined ? `${s.maxRisk}` : null);
        set("decision", dec.action ? `${dec.action} (${dec.urgency}) ${dec.reason}` : null);
        set("decisionConfidence", dec.confidence);
        set("switches", dec.switches !== undefined
            ? `${dec.switches} / suppressed ${dec.suppressedSwitches}`
            : null);
        set("corridor", cor.left !== undefined
            ? `${cor.left}-${cor.right} far ${cor.farBoundary} clutter ${cor.clutter}`
            : null);

        set("healthState", hp.state ? `${hp.state} · silent ${hp.silenceMs} ms` : null);
        set("degradation", s.degradation ? `L${s.degradation.level} ${s.degradation.label}` : null);
        set("speech", sp.available === undefined
            ? null
            : `${sp.enabled ? "on" : "muted"} · ${sp.lastVerdict} · "${sp.lastText || ""}"`);
        set("speechCounters", sp.spoken !== undefined
            ? `${sp.spoken} / ${(sp.suppressedNoChange || 0) + (sp.suppressedCooldown || 0)} (${sp.utterancesLastMinute}/min)`
            : null);
        set("recovery", rec.nextStep
            ? `${rec.nextStep} · attempts ${rec.cycleAttempts}${rec.exhausted ? " · EXHAUSTED" : ""}`
            : null);
        set("memory", hp.heapUsedMb !== null && hp.heapUsedMb !== undefined
            ? `${hp.heapUsedMb} MB / +${hp.heapGrowthMb} MB (${hp.memoryWarnings} warn)`
            : "unavailable");
        set("buffers", s.pipeline?.pool
            ? `${s.pipeline.pool.available}/${s.pipeline.pool.poolSize} free, ${Math.round(s.pipeline.pool.bytes / 1024)} KB`
            : null);
        set("coords", s.coords
            ? `${s.coords.video}→${s.coords.display} ${s.coords.objectFit}${s.coords.mirrored ? " mirrored" : ""} crop ${s.coords.croppedFraction}`
            : null);

        return out;
    }
}

export default PerformanceDashboard;
