/**
 * Object class catalogue for walking-guidance reasoning.
 *
 * Three things live here and nothing else:
 *   1. The COCO-80 label order, so raw model output rows can be named.
 *   2. Which classes matter for walking, and how much.
 *   3. Coarse physical priors used to sanity-check the monocular distance
 *      estimate. These are deliberately rough; see `docs/` for why we never
 *      claim a metric distance.
 *
 * Pure data + pure functions. No DOM access, so this module is importable in
 * Node for the automated tests.
 */

/** COCO 2017 class order, as emitted by YOLO/EfficientDet COCO exports. */
export const COCO_CLASSES = Object.freeze([
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]);

/**
 * Hazard tiers.
 *
 * VEHICLE   - can arrive fast and causes severe injury. Highest priority.
 * PERSON    - moves unpredictably, common, must be handled well.
 * STRUCTURE - fixed infrastructure at body height.
 * FURNITURE - trip/collide hazards at knee-to-torso height.
 * BAGGAGE   - low ground-level trip hazards.
 * IGNORED   - present in COCO but irrelevant to an immediate walking decision.
 */
export const HazardTier = Object.freeze({
    VEHICLE: "VEHICLE",
    PERSON: "PERSON",
    ANIMAL: "ANIMAL",
    STRUCTURE: "STRUCTURE",
    FURNITURE: "FURNITURE",
    BAGGAGE: "BAGGAGE",
    UNKNOWN: "UNKNOWN",
    IGNORED: "IGNORED"
});

/**
 * `priority` multiplies the risk score (see navigation/riskEngine.js).
 * `groundAnchored` marks classes whose bottom edge reliably touches the floor,
 * which makes the vertical-position distance cue trustworthy.
 * `typicalHeightM` is a very coarse real-world height prior.
 */
const TIER_PROPERTIES = Object.freeze({
    [HazardTier.VEHICLE]: { priority: 1.45, groundAnchored: true, typicalHeightM: 1.6 },
    [HazardTier.PERSON]: { priority: 1.30, groundAnchored: true, typicalHeightM: 1.7 },
    [HazardTier.ANIMAL]: { priority: 1.20, groundAnchored: true, typicalHeightM: 0.6 },
    [HazardTier.STRUCTURE]: { priority: 1.20, groundAnchored: true, typicalHeightM: 1.0 },
    [HazardTier.FURNITURE]: { priority: 1.10, groundAnchored: true, typicalHeightM: 0.85 },
    [HazardTier.BAGGAGE]: { priority: 1.05, groundAnchored: true, typicalHeightM: 0.45 },
    [HazardTier.UNKNOWN]: { priority: 1.00, groundAnchored: false, typicalHeightM: 0.9 },
    [HazardTier.IGNORED]: { priority: 0.0, groundAnchored: false, typicalHeightM: 0.3 }
});

const CLASS_TIERS = new Map(Object.entries({
    // Vehicles
    car: HazardTier.VEHICLE,
    bus: HazardTier.VEHICLE,
    truck: HazardTier.VEHICLE,
    motorcycle: HazardTier.VEHICLE,
    bicycle: HazardTier.VEHICLE,
    train: HazardTier.VEHICLE,
    // People
    person: HazardTier.PERSON,
    // Animals that realistically share a footpath
    dog: HazardTier.ANIMAL,
    cat: HazardTier.ANIMAL,
    horse: HazardTier.ANIMAL,
    cow: HazardTier.ANIMAL,
    // Fixed structure
    "traffic light": HazardTier.STRUCTURE,
    "stop sign": HazardTier.STRUCTURE,
    "fire hydrant": HazardTier.STRUCTURE,
    "parking meter": HazardTier.STRUCTURE,
    bench: HazardTier.STRUCTURE,
    // Furniture / indoor obstacles
    chair: HazardTier.FURNITURE,
    couch: HazardTier.FURNITURE,
    bed: HazardTier.FURNITURE,
    "dining table": HazardTier.FURNITURE,
    "potted plant": HazardTier.FURNITURE,
    toilet: HazardTier.FURNITURE,
    sink: HazardTier.FURNITURE,
    refrigerator: HazardTier.FURNITURE,
    oven: HazardTier.FURNITURE,
    tv: HazardTier.FURNITURE,
    // Ground-level trip hazards
    backpack: HazardTier.BAGGAGE,
    suitcase: HazardTier.BAGGAGE,
    handbag: HazardTier.BAGGAGE,
    umbrella: HazardTier.BAGGAGE,
    skateboard: HazardTier.BAGGAGE,
    "sports ball": HazardTier.BAGGAGE,
    bottle: HazardTier.BAGGAGE,
    vase: HazardTier.BAGGAGE
}));

/**
 * Labels we never want to spend decision bandwidth on. Small desk objects
 * dominate indoor COCO output and produce exactly the "narrates everything,
 * changes direction constantly" behaviour we are engineering away.
 */
export const IGNORED_CLASSES = new Set([
    "airplane", "boat", "bird", "sheep", "elephant", "bear", "zebra", "giraffe",
    "tie", "frisbee", "skis", "snowboard", "kite", "baseball bat",
    "baseball glove", "surfboard", "tennis racket", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "toaster",
    "book", "clock", "scissors", "teddy bear", "hair drier", "toothbrush"
]);

/**
 * A synthetic label used when geometry says "something solid is in the way"
 * but the detector has no confident class for it. Requirement 64/65: for
 * navigation, geometry outranks identity.
 */
export const UNKNOWN_OBSTACLE_LABEL = "obstacle";

/** @returns {string} the HazardTier for a label. */
export function tierForLabel(label) {
    if (!label) return HazardTier.UNKNOWN;
    const key = String(label).toLowerCase();
    if (key === UNKNOWN_OBSTACLE_LABEL) return HazardTier.UNKNOWN;
    if (IGNORED_CLASSES.has(key)) return HazardTier.IGNORED;
    return CLASS_TIERS.get(key) || HazardTier.UNKNOWN;
}

/** @returns {{priority:number, groundAnchored:boolean, typicalHeightM:number, tier:string}} */
export function classProfile(label) {
    const tier = tierForLabel(label);
    return { tier, ...TIER_PROPERTIES[tier] };
}

/** Risk multiplier for a label. 0 means "drop this detection entirely". */
export function classPriority(label) {
    return classProfile(label).priority;
}

/** True when the class is worth feeding into navigation at all. */
export function isNavigationRelevant(label) {
    return tierForLabel(label) !== HazardTier.IGNORED;
}

/** True for classes that can close distance quickly on their own. */
export function isDynamicClass(label) {
    const tier = tierForLabel(label);
    return tier === HazardTier.VEHICLE || tier === HazardTier.PERSON || tier === HazardTier.ANIMAL;
}

/**
 * Indices of navigation-relevant classes within COCO_CLASSES. Passed to the
 * inference worker so class filtering happens during decode, before any
 * allocation for irrelevant boxes (requirement 63).
 * @returns {number[]}
 */
export function navigationClassIndices(classNames = COCO_CLASSES) {
    const out = [];
    for (let i = 0; i < classNames.length; i += 1) {
        if (isNavigationRelevant(classNames[i])) out.push(i);
    }
    return out;
}
