/**
 * Static integrity of the client bundle-that-isn't.
 *
 * There is no build step, so nothing catches a bad import path or a missing
 * element id until the page is opened on a phone. These checks do that job:
 *
 *  - every relative import resolves to a file that exists
 *  - every element id the JS looks up exists in index.html
 *  - every module referenced from index.html exists
 *  - the modules on the reasoning path stay free of DOM access at import time
 *  - the deployed model assets match the export manifest
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

import { CLIENT_ROOT } from "./helpers.mjs";

const JS_ROOT = path.join(CLIENT_ROOT, "js");

/** Recursively collect .js files under a directory. */
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
const INDEX_HTML = fs.readFileSync(path.join(CLIENT_ROOT, "index.html"), "utf8");

test("the client has the modules we expect and no leftovers", () => {
    assert.ok(JS_FILES.length > 25, `only found ${JS_FILES.length} modules`);

    // Modules replaced during the upgrade must be gone, not orphaned.
    const removed = [
        "vision/visionDetector.js",
        "vision/temporalTracker.js",
        "vision/barrierDetector.js",
        "vision/detectionTypes.js",
        "navigation/walkingCorridor.js",
        "navigation/spatialAnalysis.js",
        "navigation/sceneNarrative.js",
        "navigation/stateMachine.js",
        "ui/audioFirstUI.js",
        "audio/speechEngine.js",
        "camera/orientation.js",
        "utils/geometry.js",
        "utils/smoothing.js",
        "utils/timing.js"
    ];
    for (const relative of removed) {
        assert.equal(
            fs.existsSync(path.join(JS_ROOT, relative)),
            false,
            `${relative} was replaced but is still present`
        );
    }
});

test("every relative import resolves to a real file", () => {
    const problems = [];
    const importPattern = /(?:^|\n)\s*(?:import|export)[^\n;]*?from\s+["'](\.\.?\/[^"']+)["']/g;
    const dynamicPattern = /import\(\s*(?:\/\*[^*]*\*\/\s*)?["'](\.\.?\/[^"']+)["']\s*\)/g;

    for (const file of JS_FILES) {
        const source = fs.readFileSync(file, "utf8");
        for (const pattern of [importPattern, dynamicPattern]) {
            pattern.lastIndex = 0;
            let match;
            while ((match = pattern.exec(source)) !== null) {
                const resolved = path.resolve(path.dirname(file), match[1]);
                if (!fs.existsSync(resolved)) {
                    problems.push(`${path.relative(JS_ROOT, file)} -> ${match[1]}`);
                }
            }
        }
    }

    assert.deepEqual(problems, [], `unresolved imports:\n${problems.join("\n")}`);
});

/**
 * Case sensitivity.
 *
 * Windows and macOS resolve `./ui/StatusView.js` to `statusView.js` without
 * complaint. GitHub Pages does not - it returns 404. Neither the filesystem nor a
 * local `http.server` on Windows can catch that, so the check has to be explicit:
 * compare each import against the real directory entry, character for character.
 */
test("every import matches its file's exact case", () => {
    const problems = [];
    const importPattern = /(?:^|\n)\s*(?:import|export)[^\n;]*?from\s+["'](\.\.?\/[^"']+)["']/g;
    const dynamicPattern = /import\(\s*(?:\/\*[^*]*\*\/\s*)?["'](\.\.?\/[^"']+)["']\s*\)/g;

    /** True when every path segment below CLIENT_ROOT matches on-disk casing. */
    const casingMatches = (absolute) => {
        let current = CLIENT_ROOT;
        const relative = path.relative(CLIENT_ROOT, absolute);
        if (relative.startsWith("..")) return true; // outside the client tree
        for (const segment of relative.split(path.sep)) {
            const entries = fs.readdirSync(current);
            if (!entries.includes(segment)) return false;
            current = path.join(current, segment);
        }
        return true;
    };

    for (const file of JS_FILES) {
        const source = fs.readFileSync(file, "utf8");
        for (const pattern of [importPattern, dynamicPattern]) {
            pattern.lastIndex = 0;
            let match;
            while ((match = pattern.exec(source)) !== null) {
                const resolved = path.resolve(path.dirname(file), match[1]);
                if (fs.existsSync(resolved) && !casingMatches(resolved)) {
                    problems.push(`${path.relative(JS_ROOT, file)} -> ${match[1]}`);
                }
            }
        }
    }

    assert.deepEqual(problems, [], `imports whose case does not match the file:\n${problems.join("\n")}`);
});

test("index.html asset references match their files' exact case", () => {
    const problems = [];
    const refPattern = /(?:src|href)="(\.\/[^"]+)"/g;
    let match;

    while ((match = refPattern.exec(INDEX_HTML)) !== null) {
        const relative = match[1].replace(/^\.\//, "");
        let current = CLIENT_ROOT;
        for (const segment of relative.split("/")) {
            const entries = fs.readdirSync(current);
            if (!entries.includes(segment)) {
                problems.push(`${match[1]} (segment "${segment}")`);
                break;
            }
            current = path.join(current, segment);
        }
    }

    assert.deepEqual(problems, [], `index.html references with wrong case:\n${problems.join("\n")}`);
});

/**
 * GitHub Pages serves this app from a repository subpath
 * (`/AssistiveNavigationSystem_V2/`), not from the domain root. An absolute path
 * like `/models/x.onnx` resolves to the domain root and 404s there.
 */
test("no asset path is anchored to the domain root", () => {
    const problems = [];

    const htmlRefs = INDEX_HTML.matchAll(/(?:src|href)="(\/[^/][^"]*)"/g);
    for (const match of htmlRefs) problems.push(`index.html -> ${match[1]}`);

    for (const file of JS_FILES) {
        const source = fs.readFileSync(file, "utf8");
        for (const match of source.matchAll(/from\s+["'](\/[^/"'][^"']*)["']/g)) {
            problems.push(`${path.relative(JS_ROOT, file)} -> ${match[1]}`);
        }
    }

    assert.deepEqual(problems, [], `root-anchored paths break subpath hosting:\n${problems.join("\n")}`);
});

test("same-origin model URLs resolve correctly under a repository subpath", async () => {
    const { MODEL_REGISTRY, resolveModelUrl } = await import(
        `file://${path.join(JS_ROOT, "config/modelRegistry.js").replace(/\\/g, "/")}`
    );
    // The base URL a browser reports for the deployed Pages site.
    const pagesBase = "https://example.github.io/AssistiveNavigationSystem_V2/";

    for (const model of MODEL_REGISTRY) {
        if (!model.assetPath) continue;
        const url = resolveModelUrl(model, pagesBase);
        assert.ok(
            url.startsWith(`${pagesBase}models/`),
            `${model.key} resolved to ${url}, which is outside the deployed site`
        );
    }
});

test("the worker URL used by the adapter points at the worker file", () => {
    const adapter = fs.readFileSync(path.join(JS_ROOT, "vision/adapters/workerAdapter.js"), "utf8");
    const match = adapter.match(/new URL\(\s*["']([^"']+)["']\s*,\s*import\.meta\.url\s*\)/);
    assert.ok(match, "workerAdapter should build its worker URL from import.meta.url");
    const resolved = path.resolve(path.join(JS_ROOT, "vision/adapters"), match[1]);
    assert.ok(fs.existsSync(resolved), `worker not found at ${match[1]}`);
});

test("the worker is loaded as a module and only imports module-safe code", () => {
    const adapter = fs.readFileSync(path.join(JS_ROOT, "vision/adapters/workerAdapter.js"), "utf8");
    assert.match(adapter, /type:\s*["']module["']/, "the worker must be constructed as a module worker");

    const worker = fs.readFileSync(path.join(JS_ROOT, "workers/inferenceWorker.js"), "utf8");
    // `document` does not exist in a worker; touching it would throw on load.
    assert.doesNotMatch(worker, /\bdocument\./, "the worker must not reference document");
    assert.doesNotMatch(worker, /\bwindow\./, "the worker must not reference window");
});

test("every element id the JS looks up exists in index.html", () => {
    const idPattern = /getElementById\(\s*["']([^"']+)["']\s*\)/g;
    const missing = new Set();

    for (const file of JS_FILES) {
        const source = fs.readFileSync(file, "utf8");
        idPattern.lastIndex = 0;
        let match;
        while ((match = idPattern.exec(source)) !== null) {
            const id = match[1];
            if (!INDEX_HTML.includes(`id="${id}"`)) {
                missing.add(`${id} (referenced by ${path.relative(JS_ROOT, file)})`);
            }
        }
    }

    assert.deepEqual([...missing], [], `element ids referenced but not present:\n${[...missing].join("\n")}`);
});

test("index.html references only modules that exist", () => {
    const scriptPattern = /<script[^>]*src="([^"]+)"/g;
    const linkPattern = /<link[^>]*href="([^"]+)"/g;

    for (const pattern of [scriptPattern, linkPattern]) {
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(INDEX_HTML)) !== null) {
            const reference = match[1];
            if (reference.startsWith("http")) continue;
            const resolved = path.resolve(CLIENT_ROOT, reference);
            assert.ok(fs.existsSync(resolved), `index.html references missing asset ${reference}`);
        }
    }
});

test("index.html loads the app as a module", () => {
    assert.match(INDEX_HTML, /<script\s+type="module"\s+src="\.\/js\/app\.js"/);
});

test("index.html has both live regions and the critical controls", () => {
    for (const id of [
        "aria-guidance-announcer",
        "aria-status-announcer",
        "btn-start-assist",
        "btn-pause-assist",
        "btn-stop-assist",
        "btn-mute-audio",
        "camera-video-feed",
        "main-action-headline"
    ]) {
        assert.ok(INDEX_HTML.includes(`id="${id}"`), `missing #${id}`);
    }
    assert.match(INDEX_HTML, /aria-live="assertive"/, "guidance needs an assertive live region");
    assert.match(INDEX_HTML, /aria-live="polite"/, "status needs a polite live region");
});

test("the preview element is playsinline and muted so mobile autoplay works", () => {
    const videoTag = INDEX_HTML.match(/<video[^>]*>/)[0];
    assert.match(videoTag, /playsinline/);
    assert.match(videoTag, /muted/);
});

/**
 * The reasoning modules must be importable in Node, which is what makes them
 * testable at all. A stray top-level `document` reference would break that.
 */
test("reasoning modules do not touch the DOM at import time", async () => {
    const reasoningModules = [
        "config/classCatalog.js",
        "config/visionConfig.js",
        "config/navigationConfig.js",
        "config/speechConfig.js",
        "config/modelRegistry.js",
        "config/deviceProfiles.js",
        "vision/decoders.js",
        "vision/coordinateMapper.js",
        "vision/frameScheduler.js",
        "vision/frameQuality.js",
        "vision/sceneChange.js",
        "vision/tracker.js",
        "vision/backendSelector.js",
        "vision/modelCache.js",
        "perception/freeSpace.js",
        "perception/segmentationStage.js",
        "navigation/corridorModel.js",
        "navigation/riskEngine.js",
        "navigation/pathScoring.js",
        "navigation/decisionEngine.js",
        "audio/speechPolicy.js",
        "audio/speechQueue.js",
        "health/visionHealthMonitor.js",
        "health/recoveryManager.js",
        "health/degradation.js",
        "testing/pipelineHarness.js",
        "testing/scenarios.js"
    ];

    for (const relative of reasoningModules) {
        const url = `file://${path.join(JS_ROOT, relative).replace(/\\/g, "/")}`;
        await assert.doesNotReject(() => import(url), `${relative} failed to import in Node`);
    }
});

/**
 * Loading app.js exercises the entire dependency graph, which is the only static
 * check that catches a mistyped export name. Without it, `import { Foo }` against
 * a module that exports `Bar` is a runtime error discovered on a phone.
 */
test("the orchestrator's whole dependency graph loads", async () => {
    const url = `file://${path.join(JS_ROOT, "app.js").replace(/\\/g, "/")}`;
    const module = await import(url);
    assert.equal(typeof module.NavigationApp, "function");
});

test("browser-only modules import cleanly outside a browser", async () => {
    // These touch the DOM inside their constructors, which is fine, but must not
    // do so at module scope.
    const browserModules = [
        "camera/cameraManager.js",
        "ui/statusView.js",
        "ui/assistanceControls.js",
        "ui/debugOverlay.js",
        "ui/performanceDashboard.js",
        "ui/benchmarkScreen.js",
        "ui/demoMode.js",
        "ui/scenarioRunner.js",
        "audio/speechManager.js",
        "audio/voiceManager.js",
        "vision/framePipeline.js",
        "vision/inferenceClient.js",
        "vision/adapters/workerAdapter.js",
        "vision/adapters/mediapipeAdapter.js",
        "vision/adapters/tfjsCocoSsdAdapter.js"
    ];

    for (const relative of browserModules) {
        const url = `file://${path.join(JS_ROOT, relative).replace(/\\/g, "/")}`;
        await assert.doesNotReject(() => import(url), `${relative} touches the DOM at import time`);
    }
});

test("the deployed model assets match the export manifest", () => {
    const modelsDir = path.join(CLIENT_ROOT, "models");
    const manifestPath = path.join(modelsDir, "manifest.json");
    assert.ok(fs.existsSync(manifestPath), "models/manifest.json is missing; run scripts/export_web_models.py");

    const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    let artifacts = 0;

    for (const candidate of manifest.candidates) {
        if (candidate.skipped) continue;
        for (const artifact of candidate.artifacts) {
            const file = path.join(modelsDir, artifact.file);
            assert.ok(fs.existsSync(file), `manifest lists ${artifact.file} but it is not deployed`);
            assert.equal(
                fs.statSync(file).size,
                artifact.bytes,
                `${artifact.file} size does not match the manifest`
            );
            artifacts += 1;
        }
    }

    assert.ok(artifacts >= 2, `only ${artifacts} model artifacts are deployed`);
});

test("every same-origin model in the registry is actually deployed", async () => {
    const { MODEL_REGISTRY } = await import(
        `file://${path.join(JS_ROOT, "config/modelRegistry.js").replace(/\\/g, "/")}`
    );

    for (const model of MODEL_REGISTRY) {
        if (!model.assetPath) continue;
        const file = path.join(CLIENT_ROOT, model.assetPath);
        assert.ok(fs.existsSync(file), `${model.key} declares ${model.assetPath} which is not present`);
        assert.equal(
            fs.statSync(file).size,
            model.approxBytes,
            `${model.key} approxBytes is stale; the cache validates against it`
        );
    }
});

test("the static client declares itself as ES modules for the test runner", () => {
    const pkgPath = path.join(CLIENT_ROOT, "package.json");
    assert.ok(fs.existsSync(pkgPath), "static/package.json is required for node --test to resolve these sources");
    const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
    assert.equal(pkg.type, "module");
});

test("the GitHub Pages workflow deploys the directory these files live in", () => {
    const workflow = fs.readFileSync(
        path.resolve(CLIENT_ROOT, "../../../../.github/workflows/deploy-pages.yml"),
        "utf8"
    );
    assert.match(workflow, /src\/assistive_navigation\/web\/static/);
});
