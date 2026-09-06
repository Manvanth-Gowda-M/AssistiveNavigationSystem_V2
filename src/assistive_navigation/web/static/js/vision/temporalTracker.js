/**
 * Temporal Tracker Module
 * Performs multi-frame object association, ID persistence, velocity tracking,
 * exponential smoothing, and temporal confirmation gating.
 */

import { GeometryUtils } from "../utils/geometry.js";
import { SmoothingUtils } from "../utils/smoothing.js";
import { VisionConfig } from "../config/visionConfig.js";

export class TrackedObjectState {
    constructor(id, initialDetection) {
        this.id = id;
        this.label = initialDetection.label;
        this.confidence = initialDetection.confidence;
        this.boundingBox = [...initialDetection.boundingBox];
        this.areaNorm = initialDetection.areaNorm;
        this.centerX = initialDetection.centerX;
        this.centerY = initialDetection.centerY;
        
        this.age = 1;
        this.framesConfirmed = 1;
        this.lastSeen = 0; // Frames since last matched
        this.lastTimestamp = initialDetection.timestamp || Date.now();
        this.areaGrowthRate = 0.0;
        this.isApproaching = false;
        this.isConfirmed = false;
        
        // Rolling history (last 15 frames)
        this.history = [{
            box: this.boundingBox,
            area: this.areaNorm,
            time: this.lastTimestamp
        }];
    }

    update(detection, timestamp) {
        const dtSeconds = Math.max(0.016, (timestamp - this.lastTimestamp) / 1000);
        this.lastTimestamp = timestamp;
        this.lastSeen = 0;
        this.age += 1;
        this.framesConfirmed += 1;

        // Smooth position & area
        this.boundingBox = SmoothingUtils.smoothBox(detection.boundingBox, this.boundingBox, 0.45);
        this.areaNorm = (this.boundingBox[2] - this.boundingBox[0]) * (this.boundingBox[3] - this.boundingBox[1]);
        this.centerX = (this.boundingBox[0] + this.boundingBox[2]) / 2;
        this.centerY = (this.boundingBox[1] + this.boundingBox[3]) / 2;
        this.confidence = SmoothingUtils.expSmooth(detection.confidence, this.confidence, 0.4);

        // Compute approach rate
        const growthRate = SmoothingUtils.computeAreaGrowthRate(this.areaNorm, this.history[this.history.length - 1].area, dtSeconds);
        this.areaGrowthRate = SmoothingUtils.expSmooth(growthRate, this.areaGrowthRate, 0.35);
        this.isApproaching = this.areaGrowthRate > 0.05;

        // Check confirmation gate
        if (this.framesConfirmed >= VisionConfig.temporalConfirmFrames) {
            this.isConfirmed = true;
        }

        // Store history
        this.history.push({
            box: this.boundingBox,
            area: this.areaNorm,
            time: timestamp
        });
        if (this.history.length > 15) {
            this.history.shift();
        }
    }

    coast() {
        this.lastSeen += 1;
        this.age += 1;
    }
}

export class TemporalTracker {
    constructor() {
        this.nextTrackId = 1;
        this.tracks = new Map(); // id -> TrackedObjectState
    }

    /**
     * Associates new detections with existing tracks using IoU & Center distance.
     * Returns list of confirmed and active TrackedObjectState instances.
     */
    update(detections, timestamp = Date.now()) {
        const unmatchedTracks = new Set(this.tracks.keys());
        const unmatchedDetections = new Set(detections.map((_, i) => i));

        // 1. Match based on IoU & Spatial proximity
        for (const [trackId, track] of this.tracks.entries()) {
            let bestIoU = 0.20; // Minimum matching IoU threshold
            let bestMatchIdx = -1;

            for (const detIdx of unmatchedDetections) {
                const det = detections[detIdx];
                if (det.label === track.label || (VisionConfig.highPriorityLabels.has(det.label) && VisionConfig.highPriorityLabels.has(track.label))) {
                    const iou = GeometryUtils.calculateIoU(track.boundingBox, det.boundingBox);
                    if (iou > bestIoU) {
                        bestIoU = iou;
                        bestMatchIdx = detIdx;
                    }
                }
            }

            if (bestMatchIdx !== -1) {
                track.update(detections[bestMatchIdx], timestamp);
                unmatchedTracks.delete(trackId);
                unmatchedDetections.delete(bestMatchIdx);
            }
        }

        // 2. Age / coast unmatched existing tracks
        for (const trackId of unmatchedTracks) {
            const track = this.tracks.get(trackId);
            track.coast();
            if (track.lastSeen > VisionConfig.trackTimeoutFrames) {
                this.tracks.delete(trackId);
            }
        }

        // 3. Register new tracks for remaining unmatched detections
        for (const detIdx of unmatchedDetections) {
            const det = detections[detIdx];
            const newTrack = new TrackedObjectState(this.nextTrackId++, det);
            this.tracks.set(newTrack.id, newTrack);
        }

        // Return array of active tracks
        return Array.from(this.tracks.values());
    }

    clear() {
        this.tracks.clear();
        this.nextTrackId = 1;
    }
}
