/**
 * Developer overlay.
 *
 * Draws in DISPLAY space, obtained from the coordinate mapper, so what you see
 * is a direct check on the coordinate transforms. If a box does not sit on the
 * object, the mapping is wrong - which is exactly the bug class this overlay
 * exists to catch (requirements 42, 43).
 *
 * Rendering is throttled and only runs when visible; the overlay must never be
 * the reason the frame rate drops.
 */

import { Action } from "../navigation/decisionEngine.js";

const COLOURS = {
    corridor: "rgba(0, 229, 255, 0.75)",
    corridorFill: "rgba(0, 229, 255, 0.07)",
    veryNear: "#ff1744",
    near: "#ff9100",
    medium: "#ffd600",
    far: "#00e676",
    unconfirmed: "rgba(255,255,255,0.45)",
    emergent: "#ff00e5",
    freeOpen: "rgba(0, 230, 118, 0.55)",
    freeBlocked: "rgba(255, 23, 68, 0.65)",
    text: "#e8f4ff",
    textBg: "rgba(6, 12, 22, 0.82)"
};

const DISTANCE_COLOUR = {
    VERY_NEAR: COLOURS.veryNear,
    NEAR: COLOURS.near,
    MEDIUM: COLOURS.medium,
    FAR: COLOURS.far,
    UNKNOWN: COLOURS.unconfirmed
};

export class DebugOverlay {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {{mapper:object, minIntervalMs?:number}} opts
     */
    constructor(canvas, { mapper, minIntervalMs = 60 } = {}) {
        this.canvas = canvas;
        this.mapper = mapper;
        this.minIntervalMs = minIntervalMs;
        this.ctx = canvas ? canvas.getContext("2d", { alpha: true }) : null;
        this.isVisible = false;
        this._lastRenderMs = 0;
        this._cssWidth = 0;
        this._cssHeight = 0;
        this._box = [0, 0, 0, 0];
        this.renders = 0;
    }

    setVisible(visible) {
        this.isVisible = visible;
        if (this.canvas) this.canvas.style.display = visible ? "block" : "none";
        if (!visible) this.clear();
        return this.isVisible;
    }

    toggle() {
        return this.setVisible(!this.isVisible);
    }

    clear() {
        if (this.ctx) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /** Match the backing store to the CSS box, accounting for device pixels. */
    _syncSize() {
        const cssWidth = this.mapper.displayWidth;
        const cssHeight = this.mapper.displayHeight;
        if (!cssWidth || !cssHeight) return false;

        if (cssWidth === this._cssWidth && cssHeight === this._cssHeight) return true;

        // Cap the backing store: on a 3x display a full-resolution overlay costs
        // real time for no readability gain.
        const dpr = Math.min(2, (typeof window !== "undefined" ? window.devicePixelRatio : 1) || 1);
        this.canvas.width = Math.round(cssWidth * dpr);
        this.canvas.height = Math.round(cssHeight * dpr);
        this.canvas.style.width = `${cssWidth}px`;
        this.canvas.style.height = `${cssHeight}px`;
        this._cssWidth = cssWidth;
        this._cssHeight = cssHeight;
        return true;
    }

    /**
     * @param {object} frame
     * @param {Array} frame.tracks
     * @param {object} frame.corridor CorridorModel
     * @param {object} frame.freeSpace
     * @param {object} frame.decision
     * @param {object} frame.pathScores
     * @param {number} frame.nowMs
     */
    render(frame) {
        if (!this.isVisible || !this.ctx) return;
        if (frame.nowMs - this._lastRenderMs < this.minIntervalMs) return;
        if (!this._syncSize()) return;
        this._lastRenderMs = frame.nowMs;
        this.renders += 1;

        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        ctx.clearRect(0, 0, w, h);

        this._drawFreeSpace(ctx, w, h, frame.freeSpace);
        this._drawCorridor(ctx, w, h, frame.corridor);
        this._drawTracks(ctx, w, h, frame.tracks);
        this._drawDecisionBanner(ctx, w, h, frame.decision, frame.pathScores);
    }

    /**
     * Free-space grid along the bottom edge.
     *
     * Drawn in navigation space mapped through the visible region, which is the
     * space the decision engine reasons in. Reading it against the scene is the
     * fastest way to tell whether free-space estimation is behaving.
     */
    _drawFreeSpace(ctx, w, h, freeSpace) {
        if (!freeSpace?.freeDepth) return;
        const depth = freeSpace.freeDepth;
        const n = depth.length;
        const barHeight = Math.max(10, h * 0.045);
        const y = h - barHeight - 2;
        const colWidth = w / n;

        for (let i = 0; i < n; i += 1) {
            const d = depth[i];
            const filled = barHeight * d;
            ctx.fillStyle = d >= 0.42 ? COLOURS.freeOpen : COLOURS.freeBlocked;
            ctx.fillRect(i * colWidth + 0.5, y + (barHeight - filled), colWidth - 1, filled);
        }

        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.lineWidth = 1;
        ctx.strokeRect(0.5, y, w - 1, barHeight);

        // Widest passable gap marker.
        if (freeSpace.widestGap?.widthFraction > 0) {
            const gx1 = (freeSpace.widestGap.startCol / n) * w;
            const gx2 = (freeSpace.widestGap.endCol / n) * w;
            ctx.strokeStyle = freeSpace.passable ? "#00e676" : "#ff9100";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(gx1, y - 5);
            ctx.lineTo(gx2, y - 5);
            ctx.stroke();
        }
    }

    _drawCorridor(ctx, w, h, corridor) {
        if (!corridor) return;
        // Corridor bounds are navigation coordinates. Navigation space is the
        // visible region, and the overlay covers exactly the visible region, so
        // they map directly onto the canvas.
        const x1 = corridor.left * w;
        const x2 = corridor.right * w;
        const yTop = corridor.farBoundary * h;
        const yBottom = h;

        ctx.fillStyle = COLOURS.corridorFill;
        ctx.fillRect(x1, yTop, x2 - x1, yBottom - yTop);

        ctx.strokeStyle = COLOURS.corridor;
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 6]);
        ctx.strokeRect(x1, yTop, x2 - x1, yBottom - yTop);
        ctx.setLineDash([]);

        // Near boundary: inside this line, an obstacle is within a stride.
        const yNear = corridor.nearBoundary * h;
        ctx.strokeStyle = "rgba(255, 145, 0, 0.55)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x1, yNear);
        ctx.lineTo(x2, yNear);
        ctx.stroke();

        ctx.fillStyle = "rgba(255,255,255,0.55)";
        ctx.font = `${Math.max(10, Math.round(h * 0.022))}px system-ui, sans-serif`;
        ctx.fillText("CORRIDOR", x1 + 6, yTop + 14);
    }

    _drawTracks(ctx, w, h, tracks) {
        if (!tracks?.length) return;
        const fontSize = Math.max(10, Math.round(h * 0.021));
        ctx.font = `600 ${fontSize}px system-ui, sans-serif`;

        for (const track of tracks) {
            // Tracks hold navigation-space boxes, which already share the
            // overlay's coordinate frame.
            const [nx1, ny1, nx2, ny2] = track.box;
            const x = nx1 * w;
            const y = ny1 * h;
            const bw = (nx2 - nx1) * w;
            const bh = (ny2 - ny1) * h;

            const colour = track.emergent
                ? COLOURS.emergent
                : (track.confirmed ? (DISTANCE_COLOUR[track.distanceCategory] || COLOURS.far) : COLOURS.unconfirmed);

            ctx.strokeStyle = colour;
            ctx.lineWidth = track.confirmed ? 2.5 : 1.25;
            if (track.coastFrames > 0) ctx.setLineDash([4, 4]);
            ctx.strokeRect(x, y, bw, bh);
            ctx.setLineDash([]);

            // Ground-contact tick: the cue the distance estimate is built on.
            ctx.strokeStyle = colour;
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(x, y + bh);
            ctx.lineTo(x + bw, y + bh);
            ctx.stroke();

            const tag = `#${track.id} ${track.label} ${track.distanceCategory} r${track.risk}`;
            const tagWidth = ctx.measureText(tag).width + 10;
            const tagY = Math.max(0, y - fontSize - 6);
            ctx.fillStyle = COLOURS.textBg;
            ctx.fillRect(x, tagY, tagWidth, fontSize + 6);
            ctx.fillStyle = colour;
            ctx.fillText(tag, x + 5, tagY + fontSize);

            // Velocity vector, so lateral motion is visible at a glance.
            if (Math.abs(track.velocityX) > 0.03 || Math.abs(track.velocityY) > 0.03) {
                const cx = x + bw / 2;
                const cy = y + bh / 2;
                ctx.strokeStyle = colour;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(cx + track.velocityX * w * 0.4, cy + track.velocityY * h * 0.4);
                ctx.stroke();
            }
        }
    }

    _drawDecisionBanner(ctx, w, h, decision, pathScores) {
        if (!decision) return;
        const fontSize = Math.max(12, Math.round(h * 0.028));
        const pad = 8;
        const scores = pathScores
            ? `L${pathScores.left.score} C${pathScores.center.score} R${pathScores.right.score}`
            : "";
        const text = `${decision.action.replace(/_/g, " ")}  ${scores}  conf ${decision.confidence.toFixed(2)}${decision.held ? " (held)" : ""}`;

        ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
        const width = ctx.measureText(text).width + pad * 2;

        ctx.fillStyle = decision.action === Action.STOP ? "rgba(255,23,68,0.88)" : COLOURS.textBg;
        ctx.fillRect(0, 0, width, fontSize + pad * 2);
        ctx.fillStyle = COLOURS.text;
        ctx.fillText(text, pad, fontSize + pad - 2);
    }
}

export default DebugOverlay;
