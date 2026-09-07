/*
 * Boot guard — deliberately a CLASSIC script, not a module.
 *
 * Why it exists: if the module graph fails to load, or a startup step hangs, the
 * app shows "Initialising" forever with nothing to act on. That happened, and it
 * is unacceptable for a system whose whole point is to fail loudly.
 *
 * This file loads and runs before anything else, has zero dependencies, and uses
 * only ES5 syntax so that even a browser too old to parse the app still renders
 * a readable explanation. It:
 *
 *   - records every startup step on the page, not just in the console;
 *   - captures uncaught errors and unhandled rejections, including module load
 *     failures, which are otherwise completely invisible on a phone;
 *   - fires a watchdog if the app never reports ready;
 *   - offers a one-tap copyable report.
 *
 * A phone has no devtools. This is the devtools.
 */
(function () {
    "use strict";

    /**
     * The watchdog measures **lack of progress**, not total elapsed time.
     *
     * Startup legitimately takes a while: a cold model download plus WebGPU shader
     * compilation for a detector graph was measured at 24 seconds in software
     * rendering. A fixed total-time alarm fired in the middle of that and told the
     * user the app was stuck while it was working perfectly. What actually
     * indicates a hang is silence - no step reported for a long stretch.
     */
    var NO_PROGRESS_TIMEOUT_MS = 30000;
    var WATCHDOG_POLL_MS = 2000;
    var MAX_ENTRIES = 60;

    var entries = [];
    var errors = [];
    var ready = false;
    var failed = false;
    var startedAt = Date.now();
    var lastProgressAt = Date.now();
    var listEl = null;
    var panelEl = null;
    var summaryEl = null;

    function elapsed() {
        return ((Date.now() - startedAt) / 1000).toFixed(1) + "s";
    }

    function ensureDom() {
        if (listEl || !document.body) return;
        panelEl = document.getElementById("boot-panel");
        listEl = document.getElementById("boot-log");
        summaryEl = document.getElementById("boot-summary");
    }

    function render() {
        ensureDom();
        if (!listEl) return;

        var html = "";
        for (var i = 0; i < entries.length; i += 1) {
            var e = entries[i];
            html += '<li class="boot-entry boot-' + e.status + '">'
                + '<span class="boot-time">' + e.at + "</span>"
                + '<span class="boot-name">' + escapeHtml(e.name) + "</span>"
                + (e.detail ? '<span class="boot-detail">' + escapeHtml(e.detail) + "</span>" : "")
                + "</li>";
        }
        listEl.innerHTML = html;

        if (summaryEl) {
            if (failed) summaryEl.textContent = "Startup failed. Details below.";
            else if (ready) summaryEl.textContent = "Startup complete.";
            else summaryEl.textContent = "Starting up…";
        }
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    function push(name, status, detail) {
        entries.push({ name: name, status: status || "info", detail: detail || "", at: elapsed() });
        if (entries.length > MAX_ENTRIES) entries.shift();
        lastProgressAt = Date.now();
        render();
    }

    function show() {
        ensureDom();
        if (panelEl) panelEl.hidden = false;
    }

    function hide() {
        ensureDom();
        if (panelEl) panelEl.hidden = true;
    }

    /* ------------------------------------------------------- error capture */

    function recordError(kind, message, extra) {
        errors.push({ kind: kind, message: String(message), extra: extra || "" });
        push(kind, "fail", String(message));
        show();
    }

    window.addEventListener("error", function (event) {
        // A failed module or script fetch surfaces here with no message, so the
        // element's src is the only clue available.
        if (event && event.target && (event.target.src || event.target.href)) {
            recordError("Asset failed to load", event.target.src || event.target.href);
            return;
        }
        var where = event && event.filename
            ? " (" + String(event.filename).split("/").pop() + ":" + event.lineno + ")"
            : "";
        recordError("Script error", (event && event.message ? event.message : "unknown") + where);
    }, true);

    window.addEventListener("unhandledrejection", function (event) {
        var reason = event && event.reason;
        var message = reason && reason.message ? reason.message : String(reason);
        recordError("Unhandled rejection", message);
    });

    /* ---------------------------------------------------------- watchdog */

    var watchdogTimer = setInterval(function () {
        if (ready || failed) {
            clearInterval(watchdogTimer);
            return;
        }
        if (Date.now() - lastProgressAt < NO_PROGRESS_TIMEOUT_MS) return;

        clearInterval(watchdogTimer);
        failed = true;

        push("Startup stalled", "fail",
            "No progress for " + Math.round(NO_PROGRESS_TIMEOUT_MS / 1000)
            + " seconds. The last step above is where it stopped.");

        if (!window.navigationApp) {
            push("Diagnosis", "fail",
                "The application module never constructed. This usually means a script failed to load "
                + "(check the entries above) or the browser does not support ES modules.");
        } else {
            push("Diagnosis", "fail",
                "The application started but a startup step never finished. If the last step is a model "
                + "download, the network may be blocked; if it is a session build, the GPU driver may be "
                + "at fault - reload to retry on a different backend.");
        }

        push("Environment", "info", environmentLine());
        show();
    }, WATCHDOG_POLL_MS);

    function environmentLine() {
        var bits = [];
        bits.push(navigator.userAgent);
        bits.push("modules=" + ("noModule" in HTMLScriptElement.prototype ? "yes" : "no"));
        bits.push("worker=" + (typeof Worker !== "undefined" ? "yes" : "no"));
        bits.push("wasm=" + (typeof WebAssembly !== "undefined" ? "yes" : "no"));
        bits.push("webgpu=" + (navigator.gpu ? "yes" : "no"));
        bits.push("camera=" + (navigator.mediaDevices && navigator.mediaDevices.getUserMedia ? "yes" : "no"));
        bits.push("secure=" + (window.isSecureContext ? "yes" : "no"));
        bits.push("offscreen2d=" + offscreen2d());
        bits.push("speech=" + ("speechSynthesis" in window ? "yes" : "no"));
        bits.push("origin=" + location.origin);
        return bits.join(" | ");
    }

    function offscreen2d() {
        try {
            if (typeof OffscreenCanvas === "undefined") return "no";
            return new OffscreenCanvas(2, 2).getContext("2d") ? "yes" : "no-context";
        } catch (err) {
            return "error";
        }
    }

    /* ------------------------------------------------------ public surface */

    window.__vaBoot = {
        /** Record a startup step. status: "info" | "ok" | "warn" | "fail". */
        step: function (name, status, detail) {
            push(name, status, detail);
            if (status === "fail") show();
        },

        /** Startup succeeded; collapse the panel but leave it reachable. */
        done: function (detail) {
            ready = true;
            push("Vision ready", "ok", detail || "");
            hide();
        },

        /** Startup failed for a known reason. */
        fail: function (reason) {
            failed = true;
            push("Startup failed", "fail", reason || "");
            push("Environment", "info", environmentLine());
            show();
        },

        /** Plain-text report, for copying into a bug report. */
        report: function () {
            var lines = ["Vision Assistant startup report", new Date().toISOString(), environmentLine(), ""];
            for (var i = 0; i < entries.length; i += 1) {
                var e = entries[i];
                lines.push("[" + e.at + "] " + e.status.toUpperCase() + " " + e.name + (e.detail ? " — " + e.detail : ""));
            }
            if (errors.length) {
                lines.push("", "Captured errors:");
                for (var j = 0; j < errors.length; j += 1) {
                    lines.push("- " + errors[j].kind + ": " + errors[j].message);
                }
            }
            return lines.join("\n");
        },

        environment: environmentLine,
        show: show,
        hide: hide,
        get isReady() {
            return ready;
        }
    };

    /* --------------------------------------------------------- first paint */

    function init() {
        ensureDom();
        push("Page loaded", "ok", environmentLine());

        if (!window.isSecureContext) {
            recordError(
                "Insecure context",
                "The camera needs HTTPS or localhost. On " + location.protocol
                + " the browser will refuse camera access."
            );
        }
        if (!("noModule" in HTMLScriptElement.prototype)) {
            recordError("No ES module support", "This browser is too old to run the app.");
        }

        var copyBtn = document.getElementById("btn-copy-boot-report");
        if (copyBtn) {
            copyBtn.addEventListener("click", function () {
                var text = window.__vaBoot.report();
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).then(function () {
                        copyBtn.textContent = "Copied";
                    }, function () {
                        fallbackCopy(text, copyBtn);
                    });
                } else {
                    fallbackCopy(text, copyBtn);
                }
            });
        }
    }

    function fallbackCopy(text, button) {
        // Clipboard permission is commonly denied on mobile; a selectable
        // textarea is the reliable escape hatch.
        var area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "readonly");
        area.className = "boot-report-area";
        button.parentNode.appendChild(area);
        area.select();
        button.textContent = "Select and copy above";
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
}());
