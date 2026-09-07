/**
 * Deterministic pipeline harness.
 *
 * Wires the perception-independent half of the system - tracker, corridor,
 * free-space, risk, path scoring, decision, speech policy - into something that
 * can be driven by scripted detections instead of a camera.
 *
 * Used by two callers:
 *   1. The automated tests, so decision and speech behaviour is verified without
 *      a browser, a model or a phone.
 *   2. The in-app scenario runner, so an operator can replay a scenario on the
 *      actual device and watch the overlay.
 *
 * Time is injected, so a 30-second scenario runs in a millisecond and results are
 * reproducible. No DOM, no timers, no randomness.
 */

import { ObjectTracker } from "../vision/tracker.js";
import { CorridorModel } from "../navigation/corridorModel.js";
import { FreeSpaceEstimator } from "../perception/freeSpace.js";
import { RiskEngine } from "../navigation/riskEngine.js";
import { PathScoreEngine } from "../navigation/pathScoring.js";
import { DecisionEngine } from "../navigation/decisionEngine.js";
import { SpeechPolicy } from "../audio/speechPolicy.js";
import { COCO_CLASSES } from "../config/classCatalog.js";

const CLASS_INDEX = new Map(COCO_CLASSES.map((name, i) => [name, i]));

/**
 * Normalise a scripted detection into the shape the tracker expects.
 * Accepts either `box` or the legacy `boundingBox` key.
 */
export function makeDetection({ label, confidence = 0.8, box, boundingBox, visibility = 1 }) {
    const b = box || boundingBox;
    return {
        label: String(label).toLowerCase(),
        classId: CLASS_INDEX.has(String(label).toLowerCase()) ? CLASS_INDEX.get(String(label).toLowerCase()) : -1,
        confidence,
        box: [b[0], b[1], b[2], b[3]],
        visibility
    };
}

export class PipelineHarness {
    /**
     * @param {{frameIntervalMs?:number, startMs?:number, segmentation?:object|null,
     *          qualityScore?:number}} [opts]
     */
    constructor({ frameIntervalMs = 80, startMs = 10000, segmentation = null, qualityScore = 1 } = {}) {
        this.frameIntervalMs = frameIntervalMs;
        this.nowMs = startMs;
        this.segmentation = segmentation;
        this.qualityScore = qualityScore;

        this.tracker = new ObjectTracker();
        this.corridor = new CorridorModel();
        this.freeSpaceEstimator = new FreeSpaceEstimator();
        this.riskEngine = new RiskEngine();
        this.pathScoreEngine = new PathScoreEngine();
        this.decisionEngine = new DecisionEngine();
        this.speechPolicy = new SpeechPolicy();

        this.frames = [];
        this.utterances = [];
    }

    /**
     * Push one frame of detections through the whole chain.
     *
     * @param {Array} detections scripted detections (navigation coordinates)
     * @param {{trustTracks?:boolean, qualityScore?:number,
     *          qualityAllowsDirection?:boolean, qualityRequiresStop?:boolean,
     *          advanceMs?:number}} [ctx]
     * @returns {object} full per-frame record
     */
    step(detections = [], ctx = {}) {
        const {
            trustTracks = true,
            qualityScore = this.qualityScore,
            qualityAllowsDirection = true,
            qualityRequiresStop = false,
            advanceMs = this.frameIntervalMs
        } = ctx;

        this.nowMs += advanceMs;
        const normalised = detections.map(makeDetection);

        const tracked = this.tracker.update(normalised, this.nowMs, { trustTracks });

        // Corridor adapts from the *previous* frame's free space, matching the
        // live pipeline ordering: we cannot use this frame's free space to build
        // the corridor that this frame's free space depends on.
        this.corridor.update({
            freeSpace: this.freeSpaceEstimator.lastResult,
            tracks: tracked.tracks,
            reset: !trustTracks
        });

        const freeSpace = this.freeSpaceEstimator.update(
            tracked.tracks,
            this.corridor.bounds,
            this.segmentation
        );

        const risk = this.riskEngine.scoreAll(tracked.tracks, this.corridor);
        const pathScores = this.pathScoreEngine.evaluate(tracked.tracks, freeSpace, this.corridor);

        const decision = this.decisionEngine.evaluate({
            tracks: tracked.tracks,
            threats: risk.threats,
            emergent: tracked.emergent,
            pathScores,
            freeSpace,
            corridor: this.corridor,
            nowMs: this.nowMs,
            qualityScore,
            qualityAllowsDirection,
            qualityRequiresStop
        });

        const announceClear = this.decisionEngine.shouldAnnounceClear(this.nowMs);
        const speech = this.speechPolicy.evaluate({
            decision,
            nowMs: this.nowMs,
            announceClear,
            enabled: true
        });

        if (speech.speak) {
            this.speechPolicy.commit({
                text: speech.text,
                priority: speech.priority,
                emergency: speech.emergency,
                decision,
                nowMs: this.nowMs
            });
            this.utterances.push({ atMs: this.nowMs, text: speech.text, emergency: Boolean(speech.emergency) });
            if (announceClear) this.decisionEngine.markClearAnnounced();
        }

        const record = {
            nowMs: this.nowMs,
            detections: normalised.length,
            tracks: tracked.tracks.map((t) => t.snapshot()),
            confirmed: tracked.confirmed.length,
            emergent: tracked.emergent.length,
            corridor: this.corridor.metrics(),
            freeSpace: {
                left: Number(freeSpace.clearance.left.toFixed(3)),
                center: Number(freeSpace.clearance.center.toFixed(3)),
                right: Number(freeSpace.clearance.right.toFixed(3)),
                passable: freeSpace.passable,
                widestGap: Number(freeSpace.widestGap.widthFraction.toFixed(3))
            },
            pathScores: {
                left: pathScores.left.score,
                center: pathScores.center.score,
                right: pathScores.right.score,
                best: pathScores.best
            },
            maxRisk: risk.maxRisk,
            primary: risk.primary
                ? { label: risk.primary.label, risk: risk.primary.risk, distance: risk.primary.distanceCategory }
                : null,
            decision: {
                action: decision.action,
                urgency: decision.urgency,
                reason: decision.reason,
                confidence: Number(decision.confidence.toFixed(3)),
                held: Boolean(decision.held)
            },
            speech: { spoken: Boolean(speech.speak), text: speech.text || "", verdict: speech.verdict }
        };

        this.frames.push(record);
        return record;
    }

    /**
     * Run a whole scenario.
     *
     * @param {{frames:Array<Array>, repeatLast?:number, context?:object}} scenario
     * @returns {{records:Array, finalAction:string, utterances:Array, actions:string[]}}
     */
    run(scenario) {
        const { frames = [], repeatLast = 0, context = {} } = scenario;

        for (const frame of frames) {
            this.step(frame, context);
        }
        // Repeating the last frame lets a test observe steady-state behaviour:
        // whether a decision settles, and whether speech stays quiet once said.
        if (repeatLast > 0 && frames.length > 0) {
            const last = frames[frames.length - 1];
            for (let i = 0; i < repeatLast; i += 1) this.step(last, context);
        }

        return {
            records: this.frames,
            actions: this.frames.map((f) => f.decision.action),
            finalAction: this.frames.length ? this.frames[this.frames.length - 1].decision.action : null,
            utterances: this.utterances
        };
    }

    /** Number of times the action changed across the run. */
    countActionSwitches() {
        let switches = 0;
        for (let i = 1; i < this.frames.length; i += 1) {
            if (this.frames[i].decision.action !== this.frames[i - 1].decision.action) switches += 1;
        }
        return switches;
    }

    /** Number of left/right reversals - the oscillation metric. */
    countDirectionReversals() {
        const sideOf = (action) => {
            if (action.includes("LEFT")) return "L";
            if (action.includes("RIGHT")) return "R";
            return null;
        };
        let reversals = 0;
        let lastSide = null;
        for (const frame of this.frames) {
            const side = sideOf(frame.decision.action);
            if (!side) continue;
            if (lastSide && side !== lastSide) reversals += 1;
            lastSide = side;
        }
        return reversals;
    }

    /** Whether a given track id persisted across a frame range. */
    trackPersisted(trackId, fromIndex, toIndex) {
        for (let i = fromIndex; i <= toIndex && i < this.frames.length; i += 1) {
            if (!this.frames[i].tracks.some((t) => t.id === trackId)) return false;
        }
        return true;
    }

    reset() {
        this.tracker.clear();
        this.decisionEngine.reset();
        this.speechPolicy.reset();
        this.frames = [];
        this.utterances = [];
    }
}

export default PipelineHarness;
