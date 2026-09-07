/**
 * Headless browser smoke test for the web client.
 *
 * Runs the real page in a real Chromium with a synthetic camera stream and
 * asserts that startup actually completes: the module graph loads, the device is
 * profiled, a detector is selected and warmed up, the app reports ready, and live
 * inference produces decisions.
 *
 * This is the check that unit tests cannot make. The reasoning modules are all
 * unit-tested, but "does the app start in a browser" is a different question, and
 * it is the one that was failing.
 *
 * Requires `puppeteer-core` and a local Chromium-family browser:
 *     npm install --no-save puppeteer-core
 *     node tools/smoke_browser.mjs [--url http://localhost:8731/] [--headful]
 *
 * Exits non-zero on failure and prints the page's own startup trace, so a
 * failure here reads the same way it would on a phone.
 */

import { existsSync } from "node:fs";
import process from "node:process";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const argValue = (name, fallback) => {
    const index = args.indexOf(name);
    return index > -1 && args[index + 1] ? args[index + 1] : fallback;
};

const URL_UNDER_TEST = argValue("--url", "http://localhost:8731/");
const HEADFUL = args.includes("--headful");
const READY_TIMEOUT_MS = Number(argValue("--timeout", "120000"));

const BROWSER_CANDIDATES = [
    `${process.env.ProgramFiles}\\Google\\Chrome\\Application\\chrome.exe`,
    `${process.env["ProgramFiles(x86)"]}\\Google\\Chrome\\Application\\chrome.exe`,
    `${process.env.LOCALAPPDATA}\\Google\\Chrome\\Application\\chrome.exe`,
    `${process.env.ProgramFiles}\\Microsoft\\Edge\\Application\\msedge.exe`,
    `${process.env["ProgramFiles(x86)"]}\\Microsoft\\Edge\\Application\\msedge.exe`,
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
];

function findBrowser() {
    for (const candidate of BROWSER_CANDIDATES) {
        if (candidate && existsSync(candidate)) return candidate;
    }
    throw new Error("No Chromium-family browser found. Set one of the standard install paths.");
}

const failures = [];
const notes = [];

/**
 * Replace `getUserMedia` with a canvas-backed stream.
 *
 * Chrome's built-in fake device yields a tiny, featureless frame in headless
 * mode, which the frame-quality stage correctly rejects - so the run only ever
 * proved that bad input is handled. A canvas stream gives a realistic 640x480
 * feed with a floor gradient, hard-edged structure and texture, so the quality,
 * scene-change, free-space and decision stages are genuinely exercised.
 *
 * The scene deliberately contains no recognisable objects: the correct output for
 * a clear path is CONTINUE and complete silence, which is a much stronger
 * assertion than "it did not crash".
 */
async function installSyntheticCamera(page) {
    await page.evaluateOnNewDocument(() => {
        const WIDTH = 640;
        const HEIGHT = 480;

        const canvas = document.createElement("canvas");
        canvas.width = WIDTH;
        canvas.height = HEIGHT;
        const ctx = canvas.getContext("2d");
        let frame = 0;

        function draw() {
            frame += 1;

            // Floor: a vertical brightness gradient, brighter near the horizon.
            const gradient = ctx.createLinearGradient(0, HEIGHT * 0.45, 0, HEIGHT);
            gradient.addColorStop(0, "#8d8d92");
            gradient.addColorStop(1, "#5c5c62");
            ctx.fillStyle = "#c9d2dc";
            ctx.fillRect(0, 0, WIDTH, HEIGHT * 0.45);
            ctx.fillStyle = gradient;
            ctx.fillRect(0, HEIGHT * 0.45, WIDTH, HEIGHT * 0.55);

            // Hard-edged structure so the blur metric sees real gradient energy.
            ctx.fillStyle = "#3b3f47";
            ctx.fillRect(0, HEIGHT * 0.43, WIDTH, 4);
            ctx.fillStyle = "#6f7681";
            ctx.fillRect(40, HEIGHT * 0.20, 90, HEIGHT * 0.23);
            ctx.fillRect(WIDTH - 150, HEIGHT * 0.16, 110, HEIGHT * 0.27);

            // Floor texture, slowly scrolling to simulate walking forward.
            ctx.strokeStyle = "rgba(30,34,40,0.55)";
            ctx.lineWidth = 2;
            for (let i = 0; i < 9; i += 1) {
                const y = HEIGHT * 0.5 + ((i * 34 + frame * 1.6) % (HEIGHT * 0.5));
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(WIDTH, y);
                ctx.stroke();
            }

            requestAnimationFrame(draw);
        }
        draw();

        const stream = canvas.captureStream(30);
        const track = stream.getVideoTracks()[0];
        // The camera manager reads these to report geometry and facing mode.
        track.getSettings = () => ({
            width: WIDTH, height: HEIGHT, frameRate: 30, facingMode: "environment", deviceId: "synthetic"
        });
        track.getCapabilities = () => ({ width: { max: WIDTH }, height: { max: HEIGHT } });

        navigator.mediaDevices.getUserMedia = () => Promise.resolve(stream);
        navigator.mediaDevices.enumerateDevices = () => Promise.resolve([
            { kind: "videoinput", deviceId: "synthetic", label: "Synthetic rear camera", groupId: "g" }
        ]);

        window.__syntheticCamera = { width: WIDTH, height: HEIGHT };
    });
}

function check(label, condition, detail = "") {
    if (condition) {
        console.log(`  PASS  ${label}${detail ? ` — ${detail}` : ""}`);
    } else {
        failures.push(`${label}${detail ? ` — ${detail}` : ""}`);
        console.log(`  FAIL  ${label}${detail ? ` — ${detail}` : ""}`);
    }
}

async function main() {
    const executablePath = findBrowser();
    console.log(`Browser:  ${executablePath}`);
    console.log(`Target:   ${URL_UNDER_TEST}\n`);

    const browser = await puppeteer.launch({
        executablePath,
        headless: !HEADFUL,
        args: [
            // A synthetic camera so getUserMedia succeeds without hardware and
            // without a permission prompt.
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
            "--no-sandbox",
            "--disable-dev-shm-usage"
        ]
    });

    try {
        const page = await browser.newPage();
        await page.setViewport({ width: 420, height: 900, deviceScaleFactor: 2 });
        await installSyntheticCamera(page);

        const consoleErrors = [];
        const pageErrors = [];
        const failedRequests = [];

        page.on("console", (message) => {
            if (message.type() === "error") consoleErrors.push(message.text());
        });
        page.on("pageerror", (error) => pageErrors.push(String(error?.message || error)));
        page.on("requestfailed", (request) => {
            failedRequests.push(`${request.url()} :: ${request.failure()?.errorText}`);
        });

        const context = browser.defaultBrowserContext();
        await context.overridePermissions(new URL(URL_UNDER_TEST).origin, ["camera"]);

        await page.goto(URL_UNDER_TEST, { waitUntil: "domcontentloaded", timeout: 60000 });

        /* ------------------------------------------------- startup completes */

        console.log("Startup:");
        let ready = true;
        try {
            await page.waitForFunction(
                () => window.__vaBoot && window.__vaBoot.isReady === true,
                { timeout: READY_TIMEOUT_MS, polling: 250 }
            );
        } catch {
            ready = false;
        }

        const trace = await page.evaluate(() =>
            (window.__vaBoot ? window.__vaBoot.report() : "boot guard never ran")
        );

        check("app module constructed", await page.evaluate(() => Boolean(window.navigationApp)));
        check("startup reported ready", ready);

        if (!ready) {
            console.log("\n--- page startup trace ---");
            console.log(trace);
            console.log("--- end trace ---\n");
        }

        const detector = await page.evaluate(() =>
            (window.navigationApp?.inference ? window.navigationApp.inference.describe() : null)
        );
        check("a detector was selected", Boolean(detector?.ready), detector
            ? `${detector.modelLabel} on ${detector.backendLabel}, input ${detector.inputSize}`
            : "none");
        check(
            "tensor size matches the model",
            !detector || detector.inputSize === 320,
            `pipeline input ${detector?.inputSize}`
        );
        check("warm-up produced timings", Number.isFinite(detector?.warmupSteadyStateMs),
            `first ${detector?.warmupFirstPassMs} ms, steady ${detector?.warmupSteadyStateMs} ms`);

        const status = await page.$eval("#main-status-pill", (el) => el.textContent.trim());
        check("status is not an error", !/error/i.test(status), status);

        /* ---------------------------------------------------- live inference */

        if (ready) {
            console.log("\nLive inference:");
            await page.click("#btn-start-assist");

            let inferring = true;
            try {
                await page.waitForFunction(
                    () => (window.navigationApp?.scheduler?.counters.completed || 0) >= 8,
                    { timeout: 45000, polling: 250 }
                );
            } catch {
                inferring = false;
            }

            const runtime = await page.evaluate(() => {
                const app = window.navigationApp;
                return {
                    scheduler: app.scheduler.metrics(),
                    health: app.health.snapshot(),
                    tracking: app.tracker.metrics(),
                    decision: app.decisions.metrics(),
                    pipeline: app.pipeline.stats(),
                    camera: app.camera.describe(),
                    coords: app.mapper.describe(),
                    speech: app.speech.metrics(),
                    spokenTexts: app.speech.history.map((h) => h.text),
                    degradation: app.degradation.metrics(),
                    quality: app.quality.metrics(),
                    freeSpace: app.freeSpace.metrics()
                };
            });

            check("inference is producing results", inferring,
                `${runtime.scheduler.completed} completed, ${runtime.scheduler.failed} failed`);
            check("no inference failures", runtime.scheduler.failed === 0,
                `${runtime.scheduler.failed} failures: ${runtime.health.lastFailureMessage || "none"}`);
            check("health is operational", runtime.health.state !== "DEAD", runtime.health.state);
            check("camera is streaming", runtime.camera.active, runtime.camera.resolution);
            check("camera resolution is plausible", runtime.camera.resolution === "640x480",
                runtime.camera.resolution);
            check("frame quality is usable", runtime.quality.state !== "UNUSABLE",
                `${runtime.quality.state} score ${runtime.quality.score}`
                + ` (blur ${runtime.quality.blur}, contrast ${runtime.quality.contrast})`);

            // Only the worker (tensor) path uses the frame pipeline; MediaPipe and
            // TensorFlow.js preprocess internally, so zero captures is correct there.
            if (detector?.backendId?.startsWith("wasm") || detector?.backendId === "webgpu") {
                check("frames are being captured", runtime.pipeline.framesCaptured > 0,
                    `${runtime.pipeline.framesCaptured} frames via ${runtime.pipeline.surfaceKind}`);
            } else {
                notes.push(`main-thread adapter (${detector?.backendLabel}); frame pipeline unused by design`);
            }
            check("no tensor buffers leaked", runtime.pipeline.pool.available > 0,
                `${runtime.pipeline.pool.available}/${runtime.pipeline.pool.poolSize} free`);
            check("camera FPS is sane", runtime.scheduler.cameraFps > 5,
                `${runtime.scheduler.cameraFps} FPS`);
            check("degradation level is usable", runtime.degradation.level <= 2,
                `L${runtime.degradation.level} ${runtime.degradation.label}`);

            // The synthetic scene contains no obstacles, so the correct outputs are
            // CONTINUE and complete silence. This is the strongest end-to-end
            // assertion available without real objects in front of a real camera.
            check("path reported clear", runtime.decision.action === "CONTINUE",
                `${runtime.decision.action} (${runtime.decision.reason})`);
            // Lifecycle messages ("Vision assistance ready.", "Assistance
            // started.") are expected and wanted. What must not happen is any
            // *navigation* instruction on a clear path.
            const guidance = runtime.spokenTexts.filter((text) =>
                /^(Stop|Move|Obstacle|Person|Vehicle|Path clear|Narrow)/.test(text));
            check("no guidance spoken on a clear path", guidance.length === 0,
                guidance.length ? guidance.join(" | ") : `only lifecycle: ${runtime.spokenTexts.join(" | ")}`);
            // The synthetic scene has structure at the horizon, so clearance is
            // legitimately below 1.0 and varies with the scrolling floor texture.
            // What matters is that the path is substantially open and walkable.
            check("free space reads open ahead", (runtime.freeSpace?.center ?? 0) > 0.7,
                `L/C/R ${runtime.freeSpace?.left} / ${runtime.freeSpace?.center} / ${runtime.freeSpace?.right}`);
            check("a passable gap exists", runtime.freeSpace?.passable === true,
                `widest gap ${runtime.freeSpace?.widestGapWidth}`);

            notes.push(`inference FPS ${runtime.scheduler.inferenceFps}, p95 ${runtime.scheduler.latencyP95} ms`);
            notes.push(`coordinates ${runtime.coords.video} -> ${runtime.coords.display}, crop ${runtime.coords.croppedFraction}`);
            notes.push(`tracks ${runtime.tracking.active}, dropped ${runtime.scheduler.droppedFrames}`);
        }

        /* ----------------------------------------------------------- hygiene */

        console.log("\nPage hygiene:");
        const ignorableRequests = failedRequests.filter((entry) => !entry.includes("vendor/ort/"));
        check("no uncaught page errors", pageErrors.length === 0, pageErrors.join(" | "));
        check("no failed requests", ignorableRequests.length === 0, ignorableRequests.join(" | "));
        if (failedRequests.length !== ignorableRequests.length) {
            notes.push("self-hosted runtime absent (expected); loaded from CDN instead");
        }
        if (consoleErrors.length) notes.push(`console errors: ${consoleErrors.length}`);

        if (notes.length) {
            console.log("\nNotes:");
            for (const note of notes) console.log(`  - ${note}`);
        }

        if (args.includes("--trace")) {
            console.log("\n--- page startup trace ---");
            console.log(await page.evaluate(() => window.__vaBoot.report()));
            if (consoleErrors.length) {
                console.log("\n--- console errors ---");
                for (const message of consoleErrors) console.log(`  ${message}`);
            }
        }

        console.log(`\n${failures.length === 0 ? "SMOKE TEST PASSED" : `SMOKE TEST FAILED (${failures.length})`}`);
        return failures.length === 0 ? 0 : 1;
    } finally {
        await browser.close();
    }
}

main().then(
    (code) => process.exit(code),
    (error) => {
        console.error("Smoke test could not run:", error?.message || error);
        process.exit(2);
    }
);
