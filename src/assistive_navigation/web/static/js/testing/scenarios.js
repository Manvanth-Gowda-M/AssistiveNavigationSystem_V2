/**
 * Scripted perception scenarios.
 *
 * These are the regression suite for navigation behaviour. Every one of them
 * encodes a situation that either matters for safety or has burned us before.
 * They run in the automated tests (headless, deterministic) and can be replayed
 * on-device from the diagnostics panel.
 *
 * Coordinates are navigation-space normalised boxes: [x1, y1, x2, y2], y=1 at the
 * bottom of the visible region (nearest the user). Boxes are written so that the
 * ground-contact cue (`y2`) reflects the intended distance:
 *
 *   y2 >= 0.86  very near, within arm's reach
 *   y2 ~ 0.78   near, within a stride or two
 *   y2 ~ 0.62   medium
 *   y2 <= 0.50  far
 *
 * Boxes are repeated across frames because the tracker requires three consistent
 * associations before a track can drive guidance. A scenario with one frame is
 * testing exactly that gate.
 */

/** Convenience builders keep the scenario data readable. */
const box = (x1, y1, x2, y2) => [x1, y1, x2, y2];
const det = (label, b, confidence = 0.85) => ({ label, box: b, confidence });
const hold = (frame, times) => Array.from({ length: times }, () => frame.map((d) => ({ ...d })));

/** Boxes reused across scenarios. */
const CHAIR_CENTRE = box(0.36, 0.48, 0.62, 0.78);
const CHAIR_LEFT_EDGE = box(0.02, 0.50, 0.22, 0.78);
const CHAIR_RIGHT_EDGE = box(0.78, 0.50, 0.98, 0.78);
const CHAIR_LEFT_BLOCKING = box(0.00, 0.48, 0.30, 0.80);
const CHAIR_RIGHT_BLOCKING = box(0.70, 0.48, 1.00, 0.80);
const PERSON_CENTRE = box(0.38, 0.26, 0.60, 0.80);
const CRATE_CENTRE_LOW = box(0.34, 0.60, 0.66, 0.92);

export const SCENARIOS = Object.freeze([
    {
        id: "clear-path",
        name: "Clear path",
        purpose: "Nothing detected. The system must stay silent.",
        frames: hold([], 6),
        expect: { action: "CONTINUE", maxUtterances: 0 }
    },
    {
        id: "noise-spike",
        name: "Single low-confidence blip",
        purpose: "A one-frame detection below the acceptance floor must not create a track or speak.",
        frames: [
            [det("chair", CHAIR_CENTRE, 0.31)],
            [],
            [],
            []
        ],
        expect: { action: "CONTINUE", maxUtterances: 0, maxConfirmedTracks: 0 }
    },
    {
        id: "unconfirmed-detection",
        name: "Two-frame detection",
        purpose: "Confirmation needs three frames, so two must not yet drive a manoeuvre.",
        frames: [
            [det("chair", CHAIR_CENTRE)],
            [det("chair", CHAIR_CENTRE)]
        ],
        expect: { maxUtterances: 1 }
    },
    {
        id: "centre-obstacle-left-clear",
        name: "Centre obstacle, left clear, right blocked",
        purpose: "Requirement 71: must choose LEFT and never recommend the blocked side.",
        frames: hold([det("chair", CHAIR_CENTRE), det("bicycle", CHAIR_RIGHT_BLOCKING)], 8),
        expect: { side: "LEFT", notSide: "RIGHT" }
    },
    {
        id: "centre-obstacle-right-clear",
        name: "Centre obstacle, right clear, left blocked",
        purpose: "Mirror of the previous case. Catches left/right sign errors.",
        frames: hold([det("chair", CHAIR_CENTRE), det("bicycle", CHAIR_LEFT_BLOCKING)], 8),
        expect: { side: "RIGHT", notSide: "LEFT" }
    },
    {
        id: "centre-obstacle-both-blocked",
        name: "Centre obstacle, both sides blocked",
        purpose: "Requirement 32 and 71: nowhere to go means STOP.",
        frames: hold([
            det("chair", CHAIR_CENTRE),
            det("bicycle", CHAIR_LEFT_BLOCKING),
            det("suitcase", CHAIR_RIGHT_BLOCKING)
        ], 8),
        expect: { action: "STOP", speaks: "Stop" }
    },
    {
        id: "obstacle-outside-path-left",
        name: "Obstacle at the left edge, path open",
        purpose: "Something beside you is not something in front of you. Must not steer.",
        frames: hold([det("chair", CHAIR_LEFT_EDGE)], 8),
        expect: { action: "CONTINUE", maxUtterances: 1 }
    },
    {
        id: "obstacle-outside-path-right",
        name: "Obstacle at the right edge, path open",
        purpose: "Mirror of the previous case.",
        frames: hold([det("chair", CHAIR_RIGHT_EDGE)], 8),
        expect: { action: "CONTINUE", maxUtterances: 1 }
    },
    {
        id: "person-ahead",
        name: "Person standing in the path",
        purpose: "Person hazard tier, with a clear side available.",
        frames: hold([det("person", PERSON_CENTRE, 0.91)], 8),
        expect: { notAction: "CONTINUE" }
    },
    {
        id: "person-crossing",
        name: "Person crossing left to right",
        purpose: "Lateral motion must be tracked and the track must expire on exit.",
        frames: [
            [det("person", box(0.02, 0.28, 0.20, 0.80), 0.88)],
            [det("person", box(0.14, 0.28, 0.32, 0.80), 0.89)],
            [det("person", box(0.28, 0.28, 0.46, 0.80), 0.90)],
            [det("person", box(0.42, 0.28, 0.60, 0.80), 0.91)],
            [det("person", box(0.58, 0.28, 0.76, 0.80), 0.90)],
            [det("person", box(0.74, 0.28, 0.92, 0.80), 0.88)],
            // Long enough afterwards to clear the track timeout *and* the
            // decision engine's return-to-CONTINUE dwell. A shorter tail would
            // be testing the dwell rather than the track expiry.
            ...hold([], 24)
        ],
        expect: { finalAction: "CONTINUE", tracksAtEnd: 0 }
    },
    {
        id: "very-near-obstacle",
        name: "Obstacle at arm's length in the path",
        purpose: "Requirement 32: at this range a sidestep is not safe advice.",
        frames: hold([det("chair", box(0.30, 0.40, 0.70, 0.96), 0.92)], 6),
        expect: { action: "STOP" }
    },
    {
        id: "rapid-approach",
        name: "Vehicle closing fast, dead centre",
        purpose: "Approach detection must trigger STOP before the object fills the frame.",
        frames: [
            [det("car", box(0.44, 0.42, 0.58, 0.56), 0.72)],
            [det("car", box(0.41, 0.40, 0.62, 0.62), 0.80)],
            [det("car", box(0.36, 0.36, 0.68, 0.70), 0.86)],
            [det("car", box(0.28, 0.30, 0.76, 0.80), 0.90)],
            [det("car", box(0.18, 0.24, 0.86, 0.90), 0.93)]
        ],
        expect: { action: "STOP", speaks: "Stop" }
    },
    {
        id: "unknown-obstacle",
        name: "Unclassified obstacle in the path",
        purpose: "Requirements 64 and 65: geometry beats identity. Must still act.",
        frames: hold([det("obstacle", CRATE_CENTRE_LOW, 0.66)], 8),
        expect: { notAction: "CONTINUE" }
    },
    {
        id: "flicker-no-oscillation",
        name: "Detector flicker on a centre obstacle",
        purpose: "Requirement 30: intermittent detection must not produce direction flapping.",
        frames: [
            ...hold([det("chair", CHAIR_CENTRE), det("bicycle", CHAIR_RIGHT_BLOCKING)], 4),
            [det("bicycle", CHAIR_RIGHT_BLOCKING)],
            ...hold([det("chair", CHAIR_CENTRE), det("bicycle", CHAIR_RIGHT_BLOCKING)], 3),
            [det("chair", CHAIR_CENTRE)],
            ...hold([det("chair", CHAIR_CENTRE), det("bicycle", CHAIR_RIGHT_BLOCKING)], 4)
        ],
        expect: { maxDirectionReversals: 0 }
    },
    {
        id: "ambiguous-sides",
        name: "Symmetric obstacle, both sides similar",
        purpose: "A coin-flip must resolve deterministically, not alternate.",
        frames: hold([det("chair", CHAIR_CENTRE)], 12),
        expect: { maxDirectionReversals: 0 }
    },
    {
        id: "obstacle-then-clear",
        name: "Obstacle clears after a manoeuvre",
        purpose: "Requirement 38: 'Path clear' exactly once, and only after obstruction.",
        frames: [
            ...hold([det("chair", CHAIR_CENTRE), det("bicycle", CHAIR_RIGHT_BLOCKING)], 8),
            ...hold([], 40)
        ],
        expect: { finalAction: "CONTINUE", pathClearCount: 1 }
    },
    {
        id: "camera-turn",
        name: "Camera turn invalidates tracks",
        purpose: "Requirement 67: stale obstacle positions must not survive a turn.",
        frames: [
            ...hold([det("chair", CHAIR_CENTRE)], 6),
            ...hold([], 24)
        ],
        turnAtFrame: 6,
        expect: { finalAction: "CONTINUE", tracksAtEnd: 0 }
    },
    {
        id: "low-visibility",
        name: "Perception too poor for directions",
        purpose: "Requirements 40 and 69: withhold confident directions, do not invent them.",
        frames: hold([det("chair", CHAIR_CENTRE)], 6),
        context: { qualityScore: 0.3, qualityAllowsDirection: false },
        expect: { notSide: "LEFT", notSideAlt: "RIGHT" }
    },
    {
        id: "perception-unusable",
        name: "Perception unusable",
        purpose: "Must ask the user to stop rather than guess.",
        frames: hold([], 4),
        context: { qualityRequiresStop: true },
        expect: { action: "STOP", speaks: "Please stop" }
    },
    {
        id: "sustained-clear-walk",
        name: "Sixty frames of clear path",
        purpose: "Silence is a feature. A long clear walk must produce no speech at all.",
        frames: hold([], 60),
        expect: { action: "CONTINUE", maxUtterances: 0 }
    }
]);

/** Look up a scenario by id. */
export function getScenario(id) {
    return SCENARIOS.find((s) => s.id === id) || null;
}

/**
 * Long synthetic walk, generated rather than hand-written.
 *
 * Used by the stability tests: obstacles appear, approach, pass and clear on a
 * repeating cycle, so a several-thousand-frame run exercises track creation and
 * expiry many times. This is the shape of test that catches slow leaks and
 * gradually accumulating state, which manual testing never finds.
 *
 * @param {number} frameCount
 * @returns {Array<Array>} frames
 */
export function generateWalkSequence(frameCount = 600) {
    const frames = [];
    // A 120-frame cycle is about 10 seconds at 12 inference FPS: roughly the
    // pace at which a real pedestrian meets, avoids and clears one obstacle.
    // A shorter cycle makes the sequence unrealistically busy and turns speech
    // budget assertions into a test of the generator rather than the system.
    const CYCLE = 120;

    for (let i = 0; i < frameCount; i += 1) {
        const phase = i % CYCLE;
        const frame = [];

        if (phase >= 20 && phase < 70) {
            // An obstacle approaches down the centre-right, then is passed.
            const t = (phase - 20) / 50;
            const height = 0.18 + 0.55 * t;
            const y2 = 0.52 + 0.42 * t;
            const cx = 0.52 + 0.18 * t;
            const halfWidth = 0.06 + 0.14 * t;
            frame.push(det("chair", box(cx - halfWidth, y2 - height, cx + halfWidth, y2), 0.78 + 0.15 * t));
        }
        if (phase >= 50 && phase < 95) {
            // A person crosses the far field.
            const t = (phase - 50) / 45;
            const cx = 0.05 + 0.9 * t;
            frame.push(det("person", box(cx - 0.07, 0.34, cx + 0.07, 0.72), 0.84));
        }
        if (phase >= 104) {
            // A parked vehicle at the kerb, off-path.
            frame.push(det("car", box(0.00, 0.44, 0.16, 0.74), 0.80));
        }

        frames.push(frame);
    }
    return frames;
}

export default SCENARIOS;
