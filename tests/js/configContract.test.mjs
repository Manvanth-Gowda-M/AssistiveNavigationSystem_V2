/**
 * Config contract.
 *
 * This file exists because of a specific, expensive failure: `speechConfig.js`
 * was rewritten and its `defaults` key became `voice`. `voiceManager.js` still
 * read `SpeechConfig.defaults.rate`, so its constructor threw. It is constructed
 * inside `SpeechManager`, which is constructed inside `NavigationApp`, so the
 * entire application failed to start with a blank screen and no console anyone
 * could read on a phone.
 *
 * Nothing caught it. Every module imported fine - the throw was at *construction*,
 * not at import - and the reasoning tests never build the DOM-facing objects.
 *
 * So this suite checks two things the others do not:
 *   1. Every config property the source reads actually exists.
 *   2. Every subsystem can be *constructed*, not merely imported.
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

import { SpeechConfig, SpeechPriority } from "../../src/assistive_navigation/web/static/js/config/speechConfig.js";
import { VisionConfig } from "../../src/assistive_navigation/web/static/js/config/visionConfig.js";
import { NavigationConfig } from "../../src/assistive_navigation/web/static/js/config/navigationConfig.js";
import { CLIENT_ROOT } from "./helpers.mjs";

const JS_ROOT = path.join(CLIENT_ROOT, "js");

const CONFIGS = {
    SpeechConfig,
    VisionConfig,
    NavigationConfig,
    SpeechPriority
};

function collectJs(dir) {
    const out = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) out.push(...collectJs(full));
        else if (entry.name.endsWith(".js")) out.push(full);
    }
    return out;
}

const JS_FILES = collectJs(JS_ROOT);

/**
 * Strip comments and string literals before scanning.
 *
 * Necessary because the modules that were broken now *document* the old key
 * names in their header comments, and a naive scan flags those as live reads.
 */
function stripNonCode(source) {
    return source
        .replace(/\/\*[\s\S]*?\*\//g, " ")   // block comments
        .replace(/(^|[^:])\/\/[^\n]*/g, "$1") // line comments, sparing "http://"
        .replace(/`(?:\\[\s\S]|[^\\`])*`/g, "``")
        .replace(/'(?:\\.|[^\\'])*'/g, "''")
        .replace(/"(?:\\.|[^\\"])*"/g, '""');
}

/** Walk a dotted path, returning `{ found, value }`. */
function resolvePath(root, segments) {
    let current = root;
    for (const segment of segments) {
        if (current === null || current === undefined) return { found: false, value: undefined };
        if (!(segment in current)) return { found: false, value: undefined };
        current = current[segment];
    }
    return { found: true, value: current };
}

test("every config property read by the source actually exists", () => {
    const problems = [];
    // Matches `SpeechConfig.cooldowns.repeatSameMs`, up to three levels, which
    // covers every access in the codebase.
    const pattern = new RegExp(
        `\\b(${Object.keys(CONFIGS).join("|")})\\.([A-Za-z_$][\\w$]*)(?:\\.([A-Za-z_$][\\w$]*))?(?:\\.([A-Za-z_$][\\w$]*))?`,
        "g"
    );

    for (const file of JS_FILES) {
        const source = stripNonCode(fs.readFileSync(file, "utf8"));
        pattern.lastIndex = 0;
        let match;

        while ((match = pattern.exec(source)) !== null) {
            const [, configName, ...rest] = match;
            const segments = rest.filter(Boolean);
            if (segments.length === 0) continue;

            const { found } = resolvePath(CONFIGS[configName], segments);
            if (!found) {
                problems.push(`${path.relative(JS_ROOT, file)}: ${configName}.${segments.join(".")}`);
            }
        }
    }

    assert.deepEqual(
        [...new Set(problems)],
        [],
        `config properties read but not defined:\n${[...new Set(problems)].join("\n")}`
    );
});

test("every speech priority referenced by name exists", () => {
    const problems = [];
    const pattern = /SpeechPriority\.([A-Z_]+)/g;

    for (const file of JS_FILES) {
        const source = stripNonCode(fs.readFileSync(file, "utf8"));
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(source)) !== null) {
            if (!(match[1] in SpeechPriority)) {
                problems.push(`${path.relative(JS_ROOT, file)}: SpeechPriority.${match[1]}`);
            }
        }
    }

    assert.deepEqual([...new Set(problems)], [], problems.join("\n"));
});

test("every phrase referenced by name exists in the phrase library", () => {
    const problems = [];
    const pattern = /(?:SpeechConfig\.phrases|this\.cfg\.phrases|phrases)\.([A-Z_]+)/g;

    for (const file of JS_FILES) {
        const source = stripNonCode(fs.readFileSync(file, "utf8"));
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(source)) !== null) {
            if (!(match[1] in SpeechConfig.phrases)) {
                problems.push(`${path.relative(JS_ROOT, file)}: phrases.${match[1]}`);
            }
        }
    }

    assert.deepEqual([...new Set(problems)], [], problems.join("\n"));
});

test("priority ordering is coherent: STOP outranks everything", () => {
    const values = Object.values(SpeechPriority);
    assert.equal(SpeechPriority.STOP, Math.min(...values), "STOP must be the lowest number");
    assert.ok(SpeechPriority.DIRECTION < SpeechPriority.CAUTION);
    assert.ok(SpeechPriority.CAUTION < SpeechPriority.PATH_STATUS);
    assert.equal(new Set(values).size, values.length, "priorities must be distinct");
});

/* ------------------------------------------------- construction, not import */

/**
 * Minimal DOM stub.
 *
 * Just enough for the DOM-facing subsystems to construct. This is not a jsdom
 * substitute; it is a tripwire for "the constructor throws", which is the failure
 * that got shipped.
 */
function installDomStub() {
    const makeElement = () => {
        const element = {
            style: {},
            dataset: {},
            classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
            children: [],
            textContent: "",
            innerHTML: "",
            value: 0,
            hidden: false,
            checked: false,
            appendChild(child) {
                this.children.push(child);
                return child;
            },
            insertAdjacentHTML() {},
            addEventListener() {},
            removeEventListener() {},
            setAttribute() {},
            removeAttribute() {},
            getAttribute: () => null,
            getBoundingClientRect: () => ({ width: 360, height: 640 }),
            getContext: () => ({
                clearRect() {}, fillRect() {}, strokeRect() {}, drawImage() {},
                beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, fill() {}, arc() {},
                fillText() {}, measureText: () => ({ width: 40 }),
                setLineDash() {}, createLinearGradient: () => ({ addColorStop() {} }),
                getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4) })
            }),
            querySelector: () => null,
            play: () => Promise.resolve(),
            focus() {}
        };
        return element;
    };

    const document = {
        readyState: "complete",
        baseURI: "https://example.test/app/",
        body: makeElement(),
        head: makeElement(),
        documentElement: makeElement(),
        getElementById: () => makeElement(),
        querySelector: () => makeElement(),
        querySelectorAll: () => [],
        createElement: () => makeElement(),
        addEventListener() {},
        removeEventListener() {}
    };

    // `navigator` is a getter-only global in Node, so it is deliberately left
    // alone: the modules under test must cope with whatever navigator exists.
    const previous = {
        window: globalThis.window,
        document: globalThis.document,
        screen: globalThis.screen
    };

    globalThis.document = document;
    globalThis.window = {
        document,
        devicePixelRatio: 2,
        innerWidth: 360,
        innerHeight: 640,
        isSecureContext: true,
        location: { origin: "https://example.test", href: "https://example.test/app/", protocol: "https:" },
        addEventListener() {},
        removeEventListener() {},
        // Deliberately absent: speechSynthesis. The subsystems must cope with a
        // platform that has no speech at all.
        matchMedia: () => ({ matches: false, addEventListener() {} })
    };
    globalThis.screen = { width: 360, height: 640, orientation: { angle: 0, addEventListener() {} } };

    return () => {
        globalThis.window = previous.window;
        globalThis.document = previous.document;
        globalThis.screen = previous.screen;
    };
}

test("DOM-facing subsystems construct without throwing", async () => {
    const restore = installDomStub();
    try {
        const [
            { VoiceManager },
            { SpeechManager },
            { SpeechQueue, SpeechItem },
            { StatusView },
            { AssistanceControls },
            { DebugOverlay },
            { PerformanceDashboard },
            { DemoModeController },
            { CameraManager },
            { CoordinateMapper },
            { FramePipeline }
        ] = await Promise.all([
            import("../../src/assistive_navigation/web/static/js/audio/voiceManager.js"),
            import("../../src/assistive_navigation/web/static/js/audio/speechManager.js"),
            import("../../src/assistive_navigation/web/static/js/audio/speechQueue.js"),
            import("../../src/assistive_navigation/web/static/js/ui/statusView.js"),
            import("../../src/assistive_navigation/web/static/js/ui/assistanceControls.js"),
            import("../../src/assistive_navigation/web/static/js/ui/debugOverlay.js"),
            import("../../src/assistive_navigation/web/static/js/ui/performanceDashboard.js"),
            import("../../src/assistive_navigation/web/static/js/ui/demoMode.js"),
            import("../../src/assistive_navigation/web/static/js/camera/cameraManager.js"),
            import("../../src/assistive_navigation/web/static/js/vision/coordinateMapper.js"),
            import("../../src/assistive_navigation/web/static/js/vision/framePipeline.js")
        ]);

        // The exact chain that failed: VoiceManager -> SpeechManager -> app.
        const voices = new VoiceManager();
        assert.ok(Number.isFinite(voices.rate), "rate must resolve to a number");
        assert.ok(Number.isFinite(voices.pitch));
        assert.ok(Number.isFinite(voices.volume));

        const speech = new SpeechManager({ synth: null });
        assert.equal(speech.available, false, "must report unavailability rather than throw");
        assert.equal(speech.speakDecision({
            action: "CONTINUE",
            urgency: "LOW",
            reason: "path_clear",
            confidence: 1,
            primaryObstacle: null,
            emergency: false,
            signature: () => "CONTINUE|none|none"
        }).spoken, false);

        const queue = new SpeechQueue();
        queue.enqueue(new SpeechItem({ text: "Move left." }));
        assert.ok(Number.isFinite(queue.peek().priority), "a default priority must be a real number");

        const mapper = new CoordinateMapper();
        assert.doesNotThrow(() => new StatusView());
        assert.doesNotThrow(() => new AssistanceControls({}));
        assert.doesNotThrow(() => new DebugOverlay(globalThis.document.createElement(), { mapper }));
        assert.doesNotThrow(() => new PerformanceDashboard(globalThis.document.createElement()));
        assert.doesNotThrow(() => new DemoModeController({ container: globalThis.document.createElement() }));
        assert.doesNotThrow(() => new CameraManager(globalThis.document.createElement()));

        const pipeline = new FramePipeline({ inputSize: 320 });
        assert.equal(pipeline.ready, true, "the pipeline must find a drawing surface via the stub canvas");
        assert.equal(pipeline.surfaceKind, "element");
    } finally {
        restore();
    }
});

test("the speech queue prioritises correctly with the real priority values", async () => {
    const { SpeechQueue, SpeechItem } = await import(
        "../../src/assistive_navigation/web/static/js/audio/speechQueue.js"
    );

    const queue = new SpeechQueue();
    queue.enqueue(new SpeechItem({ text: "Path clear.", priority: SpeechPriority.PATH_STATUS }));
    queue.enqueue(new SpeechItem({ text: "Obstacle ahead.", priority: SpeechPriority.CAUTION }));
    queue.enqueue(new SpeechItem({ text: "Move left.", priority: SpeechPriority.DIRECTION }));

    assert.equal(queue.dequeue().text, "Move left.", "the most actionable message comes first");
    assert.equal(queue.dequeue().text, "Obstacle ahead.");
    assert.equal(queue.dequeue().text, "Path clear.");
});

test("an emergency clears everything already queued", async () => {
    const { SpeechQueue, SpeechItem } = await import(
        "../../src/assistive_navigation/web/static/js/audio/speechQueue.js"
    );

    const queue = new SpeechQueue();
    queue.enqueue(new SpeechItem({ text: "Path clear.", priority: SpeechPriority.PATH_STATUS }));
    queue.enqueue(new SpeechItem({ text: "Move left.", priority: SpeechPriority.DIRECTION }));
    queue.enqueue(new SpeechItem({ text: "Stop.", priority: SpeechPriority.STOP, isEmergency: true }));

    assert.equal(queue.length, 1);
    assert.equal(queue.peek().text, "Stop.");
});

test("an unusable priority does not corrupt the ordering", async () => {
    const { SpeechQueue, SpeechItem } = await import(
        "../../src/assistive_navigation/web/static/js/audio/speechQueue.js"
    );

    // This is what a renamed constant produced: `undefined`.
    const queue = new SpeechQueue();
    queue.enqueue(new SpeechItem({ text: "Broken.", priority: undefined }));
    queue.enqueue(new SpeechItem({ text: "Stop.", priority: SpeechPriority.STOP, isEmergency: true }));

    assert.equal(queue.length, 1, "the emergency still pre-empts");
    assert.equal(queue.peek().text, "Stop.");
});
