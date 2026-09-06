/**
 * Navigation Decision Engine Module
 * Formulates structured, safe directional actions from tracked obstacles and free-space geometry.
 * Incorporates decision hysteresis to prevent dangerous oscillating directions.
 */

import { NavigationConfig } from "../config/navigationConfig.js";
import { SpatialAnalysis } from "./spatialAnalysis.js";

export class NavigationDecision {
    constructor({
        action = "CONTINUE",
        urgency = "LOW",
        reason = "path_clear",
        confidence = 1.0,
        primaryObstacle = null
    }) {
        this.action = action; // STOP | MOVE_LEFT | MOVE_SLIGHTLY_LEFT | MOVE_RIGHT | MOVE_SLIGHTLY_RIGHT | CAUTION | CONTINUE
        this.urgency = urgency; // CRITICAL | HIGH | MEDIUM | LOW
        this.reason = reason;
        this.confidence = confidence;
        this.primaryObstacle = primaryObstacle;
        this.timestamp = Date.now();
    }
}

export class NavigationDecisionEngine {
    constructor() {
        this.lastDecision = new NavigationDecision({ action: "CONTINUE" });
        this.decisionHoldCount = 0;
    }

    /**
     * Evaluates all confirmed tracked obstacles and chooses the single safest actionable direction.
     */
    evaluate(confirmedTracks, corridor, flankClearance) {
        // 1. Identify high-risk threats inside or near the walking corridor
        const corridorThreats = confirmedTracks.filter(t => {
            const evalRes = corridor.evaluateObstacle(t.boundingBox);
            return evalRes.inCorridor || t.estimatedRisk >= 60;
        });

        // Sort threats by risk score descending
        corridorThreats.sort((a, b) => (b.estimatedRisk || 0) - (a.estimatedRisk || 0));

        // If no confirmed threats in path
        if (corridorThreats.length === 0) {
            return this.applyHysteresis(new NavigationDecision({
                action: "CONTINUE",
                urgency: "LOW",
                reason: "path_clear",
                confidence: 0.95
            }));
        }

        const primary = corridorThreats[0];
        const primaryRisk = primary.estimatedRisk || 0;
        const distCat = primary.estimatedDistance || "MEDIUM";

        // 2. EMERGENCY STOP CONDITIONS
        // Very close collision threat, rapid approach, or both flanks blocked
        if (distCat === "VERY_CLOSE" || primaryRisk >= 88 || primary.areaGrowthRate > NavigationConfig.rapidApproachThreshold) {
            return this.applyHysteresis(new NavigationDecision({
                action: "STOP",
                urgency: "CRITICAL",
                reason: distCat === "VERY_CLOSE" ? "obstacle_very_close" : "rapid_approach",
                confidence: 0.99,
                primaryObstacle: primary
            }));
        }

        // 3. DIRECTIONAL MANEUVERING / SAFE FLANK SELECTION
        const { leftClearance, rightClearance } = flankClearance;

        // If both sides are blocked (low clearance on both flanks)
        if (leftClearance < 0.40 && rightClearance < 0.40) {
            return this.applyHysteresis(new NavigationDecision({
                action: "STOP",
                urgency: "HIGH",
                reason: "both_flanks_blocked",
                confidence: 0.90,
                primaryObstacle: primary
            }));
        }

        const primaryZone = SpatialAnalysis.getDirectionZone(primary.centerX);

        // Center Obstacle: evaluate which side has more free space
        if (primaryZone === "CENTER") {
            if (rightClearance > leftClearance + 0.10) {
                const action = rightClearance > 0.70 ? "MOVE_RIGHT" : "MOVE_SLIGHTLY_RIGHT";
                return this.applyHysteresis(new NavigationDecision({
                    action,
                    urgency: "HIGH",
                    reason: "center_obstacle_right_clear",
                    confidence: 0.88,
                    primaryObstacle: primary
                }));
            } else if (leftClearance > rightClearance + 0.10) {
                const action = leftClearance > 0.70 ? "MOVE_LEFT" : "MOVE_SLIGHTLY_LEFT";
                return this.applyHysteresis(new NavigationDecision({
                    action,
                    urgency: "HIGH",
                    reason: "center_obstacle_left_clear",
                    confidence: 0.88,
                    primaryObstacle: primary
                }));
            } else {
                // Flanks roughly equal: default to slight right or caution
                return this.applyHysteresis(new NavigationDecision({
                    action: "MOVE_SLIGHTLY_RIGHT",
                    urgency: "HIGH",
                    reason: "center_obstacle_evasion",
                    confidence: 0.75,
                    primaryObstacle: primary
                }));
            }
        }

        // Left Obstacle invading corridor: steer right
        if (primaryZone === "LEFT") {
            if (rightClearance >= 0.40) {
                return this.applyHysteresis(new NavigationDecision({
                    action: "MOVE_SLIGHTLY_RIGHT",
                    urgency: "MEDIUM",
                    reason: "left_obstacle_move_right",
                    confidence: 0.85,
                    primaryObstacle: primary
                }));
            } else {
                return this.applyHysteresis(new NavigationDecision({
                    action: "CAUTION",
                    urgency: "MEDIUM",
                    reason: "left_obstacle_caution",
                    confidence: 0.80,
                    primaryObstacle: primary
                }));
            }
        }

        // Right Obstacle invading corridor: steer left
        if (primaryZone === "RIGHT") {
            if (leftClearance >= 0.40) {
                return this.applyHysteresis(new NavigationDecision({
                    action: "MOVE_SLIGHTLY_LEFT",
                    urgency: "MEDIUM",
                    reason: "right_obstacle_move_left",
                    confidence: 0.85,
                    primaryObstacle: primary
                }));
            } else {
                return this.applyHysteresis(new NavigationDecision({
                    action: "CAUTION",
                    urgency: "MEDIUM",
                    reason: "right_obstacle_caution",
                    confidence: 0.80,
                    primaryObstacle: primary
                }));
            }
        }

        return this.applyHysteresis(new NavigationDecision({
            action: "CAUTION",
            urgency: "LOW",
            reason: "general_caution",
            confidence: 0.70,
            primaryObstacle: primary
        }));
    }

    /**
     * Applies decision hysteresis to prevent rapid direction toggling.
     */
    applyHysteresis(newDecision) {
        // Critical STOP commands always override immediately without hysteresis delay
        if (newDecision.action === "STOP" || newDecision.urgency === "CRITICAL") {
            this.lastDecision = newDecision;
            this.decisionHoldCount = NavigationConfig.hysteresis.decisionHoldFrames;
            return newDecision;
        }

        // If currently holding an active directional choice and new choice is opposite
        const isOpposite = (this.lastDecision.action.includes("LEFT") && newDecision.action.includes("RIGHT")) ||
                           (this.lastDecision.action.includes("RIGHT") && newDecision.action.includes("LEFT"));

        if (isOpposite && this.decisionHoldCount > 0) {
            this.decisionHoldCount -= 1;
            return this.lastDecision; // Maintain current direction during hold period
        }

        // Otherwise accept new decision and refresh hold count
        this.lastDecision = newDecision;
        this.decisionHoldCount = NavigationConfig.hysteresis.decisionHoldFrames;
        return newDecision;
    }
}
