/**
 * Synthetic Test Scenario Runner
 * Executes 16 deterministic benchmark scenarios to verify perception, tracking, risk, decision, and speech.
 */

export const TestScenarios = [
    {
        id: "1",
        name: "1. Clear Hallway",
        description: "Empty unobstructed walking corridor.",
        frames: [
            [], [], [], []
        ],
        expectedAction: "CONTINUE",
        expectedSpeech: "Path clear."
    },
    {
        id: "2",
        name: "2. Chair Directly Ahead (Right Flank Clear)",
        description: "Chair in central walking corridor with clear right side.",
        frames: [
            [{ label: "chair", confidence: 0.85, boundingBox: [0.38, 0.45, 0.62, 0.85] }],
            [{ label: "chair", confidence: 0.88, boundingBox: [0.38, 0.45, 0.62, 0.85] }],
            [{ label: "chair", confidence: 0.89, boundingBox: [0.38, 0.45, 0.62, 0.85] }]
        ],
        expectedAction: "MOVE_SLIGHTLY_RIGHT",
        expectedSpeech: "Obstacle ahead. Move slightly right."
    },
    {
        id: "3",
        name: "3. Chair Left (Path Clear)",
        description: "Chair far on the left side outside walking path.",
        frames: [
            [{ label: "chair", confidence: 0.82, boundingBox: [0.05, 0.50, 0.22, 0.80] }],
            [{ label: "chair", confidence: 0.85, boundingBox: [0.05, 0.50, 0.22, 0.80] }]
        ],
        expectedAction: "CONTINUE",
        expectedSpeech: "Silence (Path Clear)"
    },
    {
        id: "4",
        name: "4. Chair Right (Path Clear)",
        description: "Chair far on the right side outside walking path.",
        frames: [
            [{ label: "chair", confidence: 0.80, boundingBox: [0.78, 0.50, 0.95, 0.80] }],
            [{ label: "chair", confidence: 0.83, boundingBox: [0.78, 0.50, 0.95, 0.80] }]
        ],
        expectedAction: "CONTINUE",
        expectedSpeech: "Silence (Path Clear)"
    },
    {
        id: "5",
        name: "5. Person Directly Ahead (Left Clear)",
        description: "Person in center with open left clearance.",
        frames: [
            [{ label: "person", confidence: 0.90, boundingBox: [0.40, 0.20, 0.65, 0.90] }],
            [{ label: "person", confidence: 0.92, boundingBox: [0.40, 0.20, 0.65, 0.90] }],
            [{ label: "person", confidence: 0.94, boundingBox: [0.40, 0.20, 0.65, 0.90] }]
        ],
        expectedAction: "MOVE_SLIGHTLY_RIGHT",
        expectedSpeech: "Person ahead. Move slightly right."
    },
    {
        id: "6",
        name: "6. Person Crossing Across Path",
        description: "Person moving horizontally from left to right across corridor.",
        frames: [
            [{ label: "person", confidence: 0.85, boundingBox: [0.10, 0.30, 0.28, 0.85] }],
            [{ label: "person", confidence: 0.88, boundingBox: [0.25, 0.30, 0.43, 0.85] }],
            [{ label: "person", confidence: 0.91, boundingBox: [0.45, 0.30, 0.63, 0.85] }]
        ],
        expectedAction: "MOVE_SLIGHTLY_RIGHT",
        expectedSpeech: "Person crossing ahead."
    },
    {
        id: "7",
        name: "7. Bicycle Ahead",
        description: "Parked bicycle blocking center of walking path.",
        frames: [
            [{ label: "bicycle", confidence: 0.87, boundingBox: [0.35, 0.40, 0.68, 0.88] }],
            [{ label: "bicycle", confidence: 0.89, boundingBox: [0.35, 0.40, 0.68, 0.88] }],
            [{ label: "bicycle", confidence: 0.91, boundingBox: [0.35, 0.40, 0.68, 0.88] }]
        ],
        expectedAction: "MOVE_SLIGHTLY_RIGHT",
        expectedSpeech: "Obstacle ahead. Move slightly right."
    },
    {
        id: "8",
        name: "8. Multiple Obstacles (Center Blocked + Right Blocked)",
        description: "Obstacle in center and bicycle on right flank.",
        frames: [
            [
                { label: "chair", confidence: 0.88, boundingBox: [0.35, 0.40, 0.60, 0.85] },
                { label: "bicycle", confidence: 0.82, boundingBox: [0.72, 0.45, 0.95, 0.85] }
            ],
            [
                { label: "chair", confidence: 0.90, boundingBox: [0.35, 0.40, 0.60, 0.85] },
                { label: "bicycle", confidence: 0.85, boundingBox: [0.72, 0.45, 0.95, 0.85] }
            ]
        ],
        expectedAction: "MOVE_LEFT",
        expectedSpeech: "Obstacle ahead. Move left."
    },
    {
        id: "9",
        name: "9. Flanks Blocked (Both Left and Right Blocked)",
        description: "Center obstacle and both left and right sides blocked.",
        frames: [
            [
                { label: "chair", confidence: 0.85, boundingBox: [0.35, 0.40, 0.65, 0.85] },
                { label: "box", confidence: 0.80, boundingBox: [0.05, 0.45, 0.30, 0.85] },
                { label: "suitcase", confidence: 0.80, boundingBox: [0.70, 0.45, 0.95, 0.85] }
            ],
            [
                { label: "chair", confidence: 0.89, boundingBox: [0.35, 0.40, 0.65, 0.85] },
                { label: "box", confidence: 0.83, boundingBox: [0.05, 0.45, 0.30, 0.85] },
                { label: "suitcase", confidence: 0.84, boundingBox: [0.70, 0.45, 0.95, 0.85] }
            ]
        ],
        expectedAction: "STOP",
        expectedSpeech: "Stop. Path blocked."
    },
    {
        id: "10",
        name: "10. Low Confidence Noise Spike",
        description: "Single-frame low-confidence detection below confirmation threshold.",
        frames: [
            [{ label: "chair", confidence: 0.38, boundingBox: [0.40, 0.40, 0.60, 0.80] }],
            []
        ],
        expectedAction: "CONTINUE",
        expectedSpeech: "Silence (Suppressed False Positive)"
    },
    {
        id: "11",
        name: "11. Rapidly Approaching Vehicle",
        description: "Car expanding rapidly in size directly ahead.",
        frames: [
            [{ label: "car", confidence: 0.85, boundingBox: [0.40, 0.40, 0.60, 0.65] }],
            [{ label: "car", confidence: 0.90, boundingBox: [0.30, 0.30, 0.70, 0.80] }],
            [{ label: "car", confidence: 0.95, boundingBox: [0.15, 0.15, 0.85, 0.95] }]
        ],
        expectedAction: "STOP",
        expectedSpeech: "Stop. Obstacle approaching rapidly."
    },
    {
        id: "12",
        name: "12. Very Close Obstacle",
        description: "Massive obstacle occupying whole foreground.",
        frames: [
            [{ label: "wall", confidence: 0.92, boundingBox: [0.10, 0.10, 0.90, 0.90] }],
            [{ label: "wall", confidence: 0.95, boundingBox: [0.10, 0.10, 0.90, 0.90] }]
        ],
        expectedAction: "STOP",
        expectedSpeech: "Stop. Obstacle very close."
    }
];
