/**
 * Scene Narrative & Live Environmental Description Module
 * Transforms all detected objects, proximity tiers, and spatial zones in the room
 * into rich, clear, natural spoken descriptions.
 */

import { SpatialAnalysis } from "./spatialAnalysis.js";

export class SceneNarrator {
    /**
     * Generates a natural language summary of everything present in the camera scene.
     */
    static describeScene(trackedObjects, flankClearance) {
        if (!trackedObjects || trackedObjects.length === 0) {
            return "No obstacles or objects detected. The immediate space appears open.";
        }

        const formatObjectList = (items) => {
            const counts = {};
            for (const it of items) {
                counts[it] = (counts[it] || 0) + 1;
            }
            return Object.entries(counts).map(([name, count]) => {
                return count > 1 ? `${count} ${name}s` : name;
            }).join(", ");
        };

        const leftObjects = [];
        const centerObjects = [];
        const rightObjects = [];

        for (const obj of trackedObjects) {
            const zone = SpatialAnalysis.getDirectionZone(obj.centerX);
            const dist = obj.estimatedDistance === "VERY_CLOSE" ? "very close" :
                         (obj.estimatedDistance === "CLOSE" ? "close" :
                         (obj.estimatedDistance === "MEDIUM" ? "a few steps away" : "further back"));

            const itemText = `${obj.label} (${dist})`;

            if (zone === "LEFT") leftObjects.push(obj.label);
            else if (zone === "RIGHT") rightObjects.push(obj.label);
            else centerObjects.push(itemText);
        }

        const parts = [];

        if (centerObjects.length > 0) {
            parts.push(`Ahead: ${centerObjects.join(", ")}`);
        }
        if (leftObjects.length > 0) {
            parts.push(`On your left: ${formatObjectList(leftObjects)}`);
        }
        if (rightObjects.length > 0) {
            parts.push(`On your right: ${formatObjectList(rightObjects)}`);
        }

        // Add pathway advice
        if (flankClearance) {
            if (flankClearance.leftClearance > 0.65 && flankClearance.rightClearance > 0.65) {
                parts.push("Both sides are open.");
            } else if (flankClearance.rightClearance > 0.65) {
                parts.push("Clear pathway to your right.");
            } else if (flankClearance.leftClearance > 0.65) {
                parts.push("Clear pathway to your left.");
            } else if (flankClearance.leftClearance < 0.35 && flankClearance.rightClearance < 0.35) {
                parts.push("Pathway is constricted.");
            }
        }

        return parts.join(". ");
    }

    /**
     * Short live commentary phrase when objects shift or new objects enter view.
     */
    static getLiveCommentary(activeTracks) {
        if (!activeTracks || activeTracks.length === 0) return null;

        const labels = activeTracks.map(t => {
            const zone = SpatialAnalysis.getDirectionZone(t.centerX).toLowerCase();
            return `${t.label} ${zone}`;
        });

        const unique = [...new Set(labels)].slice(0, 3);
        return unique.join(", ");
    }
}
