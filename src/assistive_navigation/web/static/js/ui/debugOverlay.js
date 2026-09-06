/**
 * Developer Debug Overlay & Scenario Simulation Studio
 * Visualizes walking corridors, bounding boxes, distance tiers, risk scores,
 * free-space meters, FPS/latency telemetry, and runs 16 testing scenarios.
 */

export class DebugOverlay {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext("2d");
        this.isVisible = false;
    }

    toggle() {
        this.isVisible = !this.isVisible;
        const panel = document.getElementById("debug-drawer-panel");
        if (panel) {
            panel.style.display = this.isVisible ? "block" : "none";
        }
        return this.isVisible;
    }

    render({
        videoWidth,
        videoHeight,
        corridor,
        trackedObjects,
        decision,
        flankClearance,
        fps,
        latencyMs,
        backend
    }) {
        if (!this.isVisible || !this.ctx) return;

        this.canvas.width = videoWidth || 640;
        this.canvas.height = videoHeight || 480;
        const w = this.canvas.width;
        const h = this.canvas.height;

        this.ctx.clearRect(0, 0, w, h);

        // 1. Draw Walking Corridor
        const cLeft = corridor.leftBoundary * w;
        const cRight = corridor.rightBoundary * w;
        const cTop = (1.0 - corridor.bottomActiveRatio) * h;

        this.ctx.fillStyle = "rgba(0, 229, 255, 0.08)";
        this.ctx.fillRect(cLeft, cTop, cRight - cLeft, h - cTop);

        this.ctx.strokeStyle = "rgba(0, 229, 255, 0.6)";
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([6, 6]);
        this.ctx.strokeRect(cLeft, cTop, cRight - cLeft, h - cTop);
        this.ctx.setLineDash([]);

        // Zone Guidelines
        this.ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.moveTo(w * 0.33, 0);
        this.ctx.lineTo(w * 0.33, h);
        this.ctx.moveTo(w * 0.67, 0);
        this.ctx.lineTo(w * 0.67, h);
        this.ctx.stroke();

        this.ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
        this.ctx.font = "12px sans-serif";
        this.ctx.fillText("LEFT ZONE", 10, 20);
        this.ctx.fillText("WALKING CORRIDOR", cLeft + 10, 20);
        this.ctx.fillText("RIGHT ZONE", w * 0.67 + 10, 20);

        // 2. Draw Tracked Objects
        for (const track of trackedObjects) {
            const [x1n, y1n, x2n, y2n] = track.boundingBox;
            const x1 = x1n * w;
            const y1 = y1n * h;
            const bw = (x2n - x1n) * w;
            const bh = (y2n - y1n) * h;

            const dist = track.estimatedDistance || "FAR";
            let color = "#00e676"; // Green
            if (dist === "VERY_CLOSE") color = "#ff1744"; // Red
            else if (dist === "CLOSE") color = "#ff9100"; // Amber
            else if (dist === "MEDIUM") color = "#00e5ff"; // Cyan

            // Bounding Box & Corner Brackets
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = track.isConfirmed ? 3 : 1.5;
            this.ctx.strokeRect(x1, y1, bw, bh);

            // Target Center Circle
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(x1 + bw / 2, y1 + bh / 2, 4, 0, Math.PI * 2);
            this.ctx.fill();

            // Label & Stats Tag
            const tag = `#${track.id} ${track.label} [${dist}] Risk:${track.estimatedRisk || 0}`;
            this.ctx.font = "bold 12px sans-serif";
            const textWidth = this.ctx.measureText(tag).width;
            this.ctx.fillStyle = "rgba(10, 15, 25, 0.85)";
            this.ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 8, 20);
            this.ctx.fillStyle = color;
            this.ctx.fillText(tag, x1 + 4, Math.max(14, y1 - 8));
        }

        // 3. Update Debug Panel Metrics
        const elFps = document.getElementById("debug-fps");
        const elLatency = document.getElementById("debug-latency");
        const elBackend = document.getElementById("debug-backend");
        const elLeftClear = document.getElementById("debug-left-clearance");
        const elRightClear = document.getElementById("debug-right-clearance");
        const elAction = document.getElementById("debug-action");

        if (elFps) elFps.textContent = `${fps} FPS`;
        if (elLatency) elLatency.textContent = `${latencyMs} ms`;
        if (elBackend) elBackend.textContent = backend;
        if (elLeftClear) elLeftClear.textContent = `${Math.round(flankClearance.leftClearance * 100)}%`;
        if (elRightClear) elRightClear.textContent = `${Math.round(flankClearance.rightClearance * 100)}%`;
        if (elAction) elAction.textContent = `${decision.action} (${decision.urgency})`;
    }
}
