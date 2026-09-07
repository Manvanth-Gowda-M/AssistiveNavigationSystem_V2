/**
 * Temporal object tracker.
 *
 * A detector answers "what is in this frame?". Navigation needs "what is in the
 * world, where is it going, and has it been there long enough to believe?".
 * That is this module's job.
 *
 * Design choices worth stating:
 *
 * - **Global greedy association, not per-track greedy.** The pre-upgrade
 *   tracker iterated tracks in map order and let each grab its best remaining
 *   detection. With two people walking past each other that reliably swaps
 *   identities. Here every plausible (track, detection) pair is scored, sorted
 *   once, and consumed best-first.
 *
 * - **IoU *or* centroid distance.** IoU alone fails on small fast objects,
 *   where consecutive boxes simply do not overlap. Centroid proximity with a
 *   velocity-predicted position covers that case.
 *
 * - **Two tiers.** Confirmed tracks (3+ consistent frames) drive normal
 *   guidance. Separately, a single frame containing a large, close, confident
 *   object in the centre is allowed to trigger an emergency response
 *   immediately - smoothing must never delay a STOP (requirements 19, 20).
 *
 * - **Coasting uses a motion model.** A lost track keeps moving at its last
 *   known velocity while its confidence decays. A lost track that freezes in
 *   place is a phantom obstacle.
 *
 * No DOM. Unit-testable.
 */

import { VisionConfig } from "../config/visionConfig.js";
import { classProfile, isDynamicClass, HazardTier, UNKNOWN_OBSTACLE_LABEL } from "../config/classCatalog.js";
import { iou } from "./decoders.js";

/** Euclidean distance between two normalised centres. */
function centroidDistance(ax, ay, bx, by) {
    const dx = ax - bx;
    const dy = ay - by;
    return Math.sqrt(dx * dx + dy * dy);
}

function expSmooth(next, prev, alpha) {
    if (prev === null || prev === undefined || !Number.isFinite(prev)) return next;
    return alpha * next + (1 - alpha) * prev;
}

export class Track {
    constructor(id, detection, timestampMs, cfg) {
        this.cfg = cfg;
        this.id = id;
        this.label = detection.label;
        this.classId = detection.classId;
        const profile = classProfile(detection.label);
        this.tier = profile.tier;
        this.groundAnchored = profile.groundAnchored;
        this.dynamic = isDynamicClass(detection.label);

        /** Smoothed box, [x1,y1,x2,y2] normalised navigation coordinates. */
        this.box = detection.box.slice();
        /** Latest unsmoothed box; used for emergency reasoning. */
        this.rawBox = detection.box.slice();

        this.confidence = detection.confidence;
        this.rawConfidence = detection.confidence;
        this.peakConfidence = detection.confidence;
        this.visibility = detection.visibility ?? 1;

        this._refreshGeometry();

        this.velocityX = 0;
        this.velocityY = 0;
        this.areaGrowthRate = 0;

        this.age = 1;
        this.hits = 1;
        this.coastFrames = 0;
        this.firstSeenMs = timestampMs;
        this.lastSeenMs = timestampMs;
        this.lastUpdateMs = timestampMs;
        this.confirmed = false;

        /** Assigned by the navigation stages; declared here so the shape is stable. */
        this.distanceCategory = "UNKNOWN";
        this.risk = 0;
        this.pathOverlap = 0;
        this.groundContact = 0;
        this.emergent = false;

        this.history = [{ t: timestampMs, area: this.area, cx: this.centerX, cy: this.centerY }];
    }

    _refreshGeometry() {
        const [x1, y1, x2, y2] = this.box;
        this.width = Math.max(0, x2 - x1);
        this.height = Math.max(0, y2 - y1);
        this.area = this.width * this.height;
        this.centerX = x1 + this.width / 2;
        this.centerY = y1 + this.height / 2;
        /** Bottom edge: the ground-contact cue. 1.0 = bottom of frame. */
        this.bottomY = y2;
    }

    /** Predicted centre one dt ahead, used for association gating. */
    predictedCentre(dtSeconds) {
        return {
            x: this.centerX + this.velocityX * dtSeconds,
            y: this.centerY + this.velocityY * dtSeconds
        };
    }

    update(detection, timestampMs) {
        const dt = Math.max(0.016, (timestampMs - this.lastUpdateMs) / 1000);
        const prevCentreX = this.centerX;
        const prevCentreY = this.centerY;
        const prevArea = this.area;

        this.rawBox[0] = detection.box[0];
        this.rawBox[1] = detection.box[1];
        this.rawBox[2] = detection.box[2];
        this.rawBox[3] = detection.box[3];

        const a = this.cfg.boxAlpha;
        this.box[0] = expSmooth(detection.box[0], this.box[0], a);
        this.box[1] = expSmooth(detection.box[1], this.box[1], a);
        this.box[2] = expSmooth(detection.box[2], this.box[2], a);
        this.box[3] = expSmooth(detection.box[3], this.box[3], a);
        this._refreshGeometry();

        this.rawConfidence = detection.confidence;
        this.confidence = expSmooth(detection.confidence, this.confidence, this.cfg.confidenceAlpha);
        if (this.rawConfidence > this.peakConfidence) this.peakConfidence = this.rawConfidence;
        this.visibility = detection.visibility ?? this.visibility;

        // A detection with a better-specified label upgrades an unknown track.
        if (this.label === UNKNOWN_OBSTACLE_LABEL && detection.label !== UNKNOWN_OBSTACLE_LABEL) {
            this.label = detection.label;
            this.classId = detection.classId;
            const profile = classProfile(detection.label);
            this.tier = profile.tier;
            this.groundAnchored = profile.groundAnchored;
            this.dynamic = isDynamicClass(detection.label);
        }

        this.velocityX = expSmooth((this.centerX - prevCentreX) / dt, this.velocityX, this.cfg.velocityAlpha);
        this.velocityY = expSmooth((this.centerY - prevCentreY) / dt, this.velocityY, this.cfg.velocityAlpha);

        // Relative area growth per second. Relative rather than absolute,
        // because a distant object doubling in size matters as much as a near
        // one growing 20%.
        const relativeGrowth = prevArea > 1e-6 ? (this.area - prevArea) / (prevArea * dt) : 0;
        this.areaGrowthRate = expSmooth(relativeGrowth, this.areaGrowthRate, this.cfg.growthAlpha);

        this.age += 1;
        this.hits += 1;
        this.coastFrames = 0;
        this.lastSeenMs = timestampMs;
        this.lastUpdateMs = timestampMs;
        if (this.hits >= this.cfg.confirmFrames) this.confirmed = true;

        this.history.push({ t: timestampMs, area: this.area, cx: this.centerX, cy: this.centerY });
        if (this.history.length > this.cfg.historyLength) this.history.shift();
    }

    /**
     * Advance an unmatched track. The box moves with the last known velocity and
     * confidence decays, so a coasted track fades out instead of lingering as a
     * stationary phantom.
     */
    coast(timestampMs) {
        const dt = Math.max(0.016, (timestampMs - this.lastUpdateMs) / 1000);
        const dx = this.velocityX * dt;
        const dy = this.velocityY * dt;

        this.box[0] += dx;
        this.box[1] += dy;
        this.box[2] += dx;
        this.box[3] += dy;
        this._refreshGeometry();

        // Roughly halve confidence over ~4 coasted frames.
        this.confidence *= 0.82;
        this.coastFrames += 1;
        this.age += 1;
        this.lastUpdateMs = timestampMs;
    }

    /** Wall-clock and frame-count expiry. */
    isExpired(timestampMs) {
        if (this.coastFrames > this.cfg.maxCoastFrames) return true;
        if (timestampMs - this.lastSeenMs > this.cfg.maxCoastMs) return true;
        // A track whose confidence has decayed below the acceptance floor is no
        // longer evidence of anything.
        if (this.confidence < VisionConfig.detection.scoreThreshold * 0.6) return true;
        return false;
    }

    /** True when the object is closing on the user. */
    get approaching() {
        return this.areaGrowthRate > 0.12;
    }

    /** True when it is closing fast enough to demand an immediate response. */
    get rapidlyApproaching() {
        return this.areaGrowthRate > 0.55;
    }

    /** Seconds this track has existed. */
    ageSeconds(timestampMs) {
        return (timestampMs - this.firstSeenMs) / 1000;
    }

    /**
     * Association stability, 0..1. Feeds the risk model so a flickering track
     * carries less weight than a rock-solid one.
     */
    get stability() {
        const hitRatio = this.age > 0 ? this.hits / this.age : 0;
        const maturity = Math.min(1, this.hits / (this.cfg.confirmFrames * 2));
        return Math.max(0, Math.min(1, 0.5 * hitRatio + 0.5 * maturity));
    }

    /** Compact snapshot for the overlay, dashboard and tests. */
    snapshot() {
        return {
            id: this.id,
            label: this.label,
            tier: this.tier,
            confidence: Number(this.confidence.toFixed(3)),
            box: this.box.slice(),
            centerX: Number(this.centerX.toFixed(4)),
            centerY: Number(this.centerY.toFixed(4)),
            bottomY: Number(this.bottomY.toFixed(4)),
            area: Number(this.area.toFixed(5)),
            velocityX: Number(this.velocityX.toFixed(4)),
            velocityY: Number(this.velocityY.toFixed(4)),
            areaGrowthRate: Number(this.areaGrowthRate.toFixed(3)),
            age: this.age,
            hits: this.hits,
            coastFrames: this.coastFrames,
            confirmed: this.confirmed,
            emergent: this.emergent,
            stability: Number(this.stability.toFixed(3)),
            distanceCategory: this.distanceCategory,
            risk: this.risk,
            pathOverlap: Number(this.pathOverlap.toFixed(3))
        };
    }
}

export class ObjectTracker {
    constructor(cfg = VisionConfig.tracking) {
        this.cfg = cfg;
        this.tracks = new Map();
        this._nextId = 1;
        this._pairs = [];
        this._confirmed = [];
        this._emergent = [];
        this._all = [];
        this.totalCreated = 0;
        this.totalExpired = 0;
        this.identitySwapGuards = 0;
    }

    /**
     * Label compatibility gate.
     *
     * Same label always matches. An unknown-geometry obstacle may match
     * anything, because it is by definition unclassified. Two *different* known
     * labels never match, even within a tier: letting a `chair` absorb a
     * `person` detection is exactly the identity swap we are guarding against.
     */
    _labelsCompatible(track, detection) {
        if (track.label === detection.label) return true;
        if (track.label === UNKNOWN_OBSTACLE_LABEL || detection.label === UNKNOWN_OBSTACLE_LABEL) return true;
        return false;
    }

    /**
     * @param {Array<{label:string, classId:number, confidence:number, box:number[], visibility?:number}>} detections
     *        boxes in normalised navigation coordinates
     * @param {number} timestampMs
     * @param {{trustTracks?:boolean, sceneChanged?:boolean}} [context]
     * @returns {{tracks:Track[], confirmed:Track[], emergent:Track[]}}
     */
    update(detections, timestampMs, context = {}) {
        const { trustTracks = true } = context;

        // A turn invalidates spatial history. Decay confidence and revoke
        // confirmation rather than deleting outright: if the objects are still
        // there they will re-confirm within a few frames.
        if (!trustTracks) this._decayAll();

        const pairs = this._pairs;
        pairs.length = 0;

        const trackList = Array.from(this.tracks.values());
        const detUsed = new Uint8Array(detections.length);
        const trackUsed = new Set();

        for (let ti = 0; ti < trackList.length; ti += 1) {
            const track = trackList[ti];
            const dtSeconds = Math.max(0.016, (timestampMs - track.lastUpdateMs) / 1000);
            const predicted = track.predictedCentre(dtSeconds);

            for (let di = 0; di < detections.length; di += 1) {
                const det = detections[di];
                if (!this._labelsCompatible(track, det)) continue;

                const overlap = iou(track.box, det.box);
                const dcx = (det.box[0] + det.box[2]) / 2;
                const dcy = (det.box[1] + det.box[3]) / 2;
                const dist = centroidDistance(predicted.x, predicted.y, dcx, dcy);

                const iouOk = overlap >= this.cfg.minMatchIou;
                const distOk = dist <= this.cfg.maxMatchCentroidDistance;
                if (!iouOk && !distOk) continue;

                // Cost blends both cues so a pair strong on either can win, but
                // a pair strong on both wins more.
                const cost = (1 - overlap) * 0.65
                    + (dist / this.cfg.maxMatchCentroidDistance) * 0.35;
                pairs.push({ ti, di, cost, overlap, dist });
            }
        }

        pairs.sort((a, b) => a.cost - b.cost);

        for (const pair of pairs) {
            if (trackUsed.has(pair.ti) || detUsed[pair.di]) {
                if (trackUsed.has(pair.ti) && !detUsed[pair.di]) this.identitySwapGuards += 1;
                continue;
            }
            trackUsed.add(pair.ti);
            detUsed[pair.di] = 1;
            trackList[pair.ti].update(detections[pair.di], timestampMs);
        }

        // Unmatched tracks coast, then expire.
        for (let ti = 0; ti < trackList.length; ti += 1) {
            if (trackUsed.has(ti)) continue;
            const track = trackList[ti];
            track.coast(timestampMs);
            if (track.isExpired(timestampMs)) {
                this.tracks.delete(track.id);
                this.totalExpired += 1;
            }
        }

        // Unmatched detections open new tracks, subject to the acceptance floor.
        for (let di = 0; di < detections.length; di += 1) {
            if (detUsed[di]) continue;
            const det = detections[di];
            if (det.confidence < VisionConfig.detection.acceptThreshold) continue;
            const track = new Track(this._nextId++, det, timestampMs, this.cfg);
            this.tracks.set(track.id, track);
            this.totalCreated += 1;
        }

        return this._collect(timestampMs);
    }

    _decayAll() {
        for (const track of this.tracks.values()) {
            track.confidence *= this.cfg.sceneChangeConfidenceDecay;
            track.confirmed = false;
            // Reset the hit count to 1, not to confirmFrames-1. If we only
            // knocked it back one frame, a track that still matches in the very
            // next frame would re-confirm immediately and the decay would have
            // achieved nothing. After a turn a track has to earn confirmation
            // again over several frames, which is the point.
            track.hits = 1;
            // Velocity measured before a turn is meaningless afterwards.
            track.velocityX = 0;
            track.velocityY = 0;
            track.areaGrowthRate = 0;
        }
    }

    _collect(timestampMs) {
        const all = this._all;
        const confirmed = this._confirmed;
        const emergent = this._emergent;
        all.length = 0;
        confirmed.length = 0;
        emergent.length = 0;

        for (const track of this.tracks.values()) {
            all.push(track);

            track.emergent = this._isEmergent(track);
            if (track.emergent) emergent.push(track);
            if (track.confirmed && track.coastFrames === 0) confirmed.push(track);
        }

        return { tracks: all, confirmed, emergent, timestampMs };
    }

    /**
     * Emergency channel (requirement 20).
     *
     * Bypasses the confirmation requirement, but not the *evidence*
     * requirement: the detection must be confident, physically large in frame,
     * and either already occupying the lower-centre region or closing fast.
     * A distant high-confidence car does not qualify; a car filling the lower
     * third does.
     */
    _isEmergent(track) {
        if (track.coastFrames > 1) return false;
        if (track.rawConfidence < VisionConfig.detection.emergencyThreshold) return false;

        const centred = track.centerX > 0.28 && track.centerX < 0.72;
        const large = track.area >= 0.14;
        const veryLarge = track.area >= 0.30;
        const low = track.bottomY >= 0.62;

        if (veryLarge && centred) return true;
        if (large && centred && low) return true;
        if (track.rapidlyApproaching && centred && track.area >= 0.06) return true;
        return false;
    }

    /** Track count, for the dashboard. */
    get count() {
        return this.tracks.size;
    }

    get confirmedCount() {
        let n = 0;
        for (const t of this.tracks.values()) if (t.confirmed) n += 1;
        return n;
    }

    metrics() {
        return {
            active: this.tracks.size,
            confirmed: this.confirmedCount,
            created: this.totalCreated,
            expired: this.totalExpired,
            identitySwapGuards: this.identitySwapGuards
        };
    }

    clear() {
        this.tracks.clear();
        this._nextId = 1;
    }
}

export { HazardTier };
