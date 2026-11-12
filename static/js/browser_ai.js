/**
 * AURA-VISION Browser AI Engine (Cloud Mode)
 * Enables real-time client-side road-user detection, multi-object tracking,
 * and 4-tier multi-frame accident detection on Live Webcam and Recorded Video
 * uploads when running statically on Vercel without a local Python backend.
 */

(function (global) {
  "use strict";

  const ROAD_USER_CLASSES = new Set(["person", "bicycle", "car", "motorcycle", "bus", "truck"]);
  const VEHICLE_CLASSES = new Set(["bicycle", "car", "motorcycle", "bus", "truck"]);

  const COLOR_LIME = "#B7D83B";
  const COLOR_AMBER = "#E5A93B";
  const COLOR_RED = "#E05252";
  const COLOR_CHARCOAL = "#191D1C";
  const COLOR_TEXT = "#F2F4EF";

  function calcIoU(b1, b2) {
    const x1 = Math.max(b1[0], b2[0]);
    const y1 = Math.max(b1[1], b2[1]);
    const x2 = Math.min(b1[2], b2[2]);
    const y2 = Math.min(b1[3], b2[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const a1 = Math.max(0, b1[2] - b1[0]) * Math.max(0, b1[3] - b1[1]);
    const a2 = Math.max(0, b2[2] - b2[0]) * Math.max(0, b2[3] - b2[1]);
    const union = a1 + a2 - inter;
    return union > 0 ? inter / union : 0;
  }

  class TrackState {
    constructor(id, bbox, className, confidence, fps) {
      this.id = id;
      this.bbox = bbox.slice();
      this.className = className;
      this.confidence = confidence;
      this.fps = fps || 15;
      this.center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
      this.history = [this.center];
      this.speedsHistory = [0];
      this.headingHistory = [0];
      this.velocity = [0, 0];
      this.smoothedSpeed = 0;
      this.maxRecentSpeed = 0;
      this.headingDeg = 0;
      this.headingDeltaDeg = 0;
      this.smoothedDeceleration = 0;
      this.isStopped = false;
      this.motionStatus = "Cruising";
      this.frameCount = 1;
      this.hits = 1;
      this.missed = 0;
      this.isConfirmed = false;
    }

    update(bbox, confidence, className, fps) {
      if (fps > 0) this.fps = fps;
      this.bbox = bbox.slice();
      const newCenter = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
      if (className) this.className = className;
      this.confidence = confidence;

      const dx = newCenter[0] - this.center[0];
      const dy = newCenter[1] - this.center[1];
      const dist = Math.hypot(dx, dy);

      this.velocity = [dx, dy];
      this.smoothedSpeed = this.frameCount === 1 ? dist : 0.6 * dist + 0.4 * this.smoothedSpeed;
      this.speedsHistory.push(this.smoothedSpeed);
      if (this.speedsHistory.length > 30) this.speedsHistory.shift();
      this.maxRecentSpeed = Math.max(...this.speedsHistory);

      const prevSpeed =
        this.speedsHistory.length >= 2 ? this.speedsHistory[this.speedsHistory.length - 2] : this.smoothedSpeed;
      const decel = Math.max(0, prevSpeed - this.smoothedSpeed);
      this.smoothedDeceleration = 0.7 * decel + 0.3 * this.smoothedDeceleration;

      if (dist > 1.5) {
        const newHeading = ((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 360;
        const prevHeading = this.headingHistory.length ? this.headingHistory[this.headingHistory.length - 1] : newHeading;
        const rawDiff = Math.abs(newHeading - prevHeading);
        this.headingDeltaDeg = Math.min(rawDiff, 360 - rawDiff);
        this.headingDeg = newHeading;
        this.headingHistory.push(newHeading);
        if (this.headingHistory.length > 30) this.headingHistory.shift();
      } else {
        this.headingDeltaDeg = 0;
      }

      this.isStopped = this.smoothedSpeed < 1.5 && this.maxRecentSpeed > 3.0;
      if (this.isStopped) this.motionStatus = "Stopped";
      else if (this.smoothedDeceleration > 3.5) this.motionStatus = "Braking";
      else if (dist > prevSpeed + 2.5) this.motionStatus = "Accelerating";
      else this.motionStatus = "Cruising";

      this.center = newCenter;
      this.history.push(newCenter);
      if (this.history.length > 24) this.history.shift();
      this.frameCount += 1;
      this.hits += 1;
      this.missed = 0;
      if (this.hits >= 2) this.isConfirmed = true;
    }

    markMissed() {
      this.missed += 1;
      this.smoothedSpeed *= 0.8;
    }

    getSpeedDisplay() {
      return `${Math.round(this.smoothedSpeed * this.fps)} px/s`;
    }
  }

  class BrowserTracker {
    constructor() {
      this.tracks = new Map();
      this.nextId = 1;
      this.totalUnique = 0;
    }

    reset() {
      this.tracks.clear();
      this.nextId = 1;
      this.totalUnique = 0;
    }

    update(detections, fps) {
      const trackIds = Array.from(this.tracks.keys());
      const unmatchedTracks = new Set(trackIds);
      const unmatchedDets = new Set(detections.map((_, idx) => idx));
      const matched = [];

      // Pass 1: IoU matching
      if (trackIds.length > 0 && detections.length > 0) {
        const pairs = [];
        for (const tid of trackIds) {
          const trk = this.tracks.get(tid);
          detections.forEach((det, di) => {
            const iou = calcIoU(trk.bbox, det.bbox);
            const score = iou + (trk.className === det.className ? 0.15 : -0.2);
            if (score >= 0.25) pairs.push({ tid, di, score });
          });
        }
        pairs.sort((a, b) => b.score - a.score);
        for (const p of pairs) {
          if (unmatchedTracks.has(p.tid) && unmatchedDets.has(p.di)) {
            matched.push([p.tid, p.di]);
            unmatchedTracks.delete(p.tid);
            unmatchedDets.delete(p.di);
          }
        }
      }

      // Pass 2: Velocity-guided centroid proximity
      for (const tid of Array.from(unmatchedTracks)) {
        const trk = this.tracks.get(tid);
        const predX = trk.center[0] + trk.velocity[0];
        const predY = trk.center[1] + trk.velocity[1];
        let bestDist = 140;
        let bestIdx = null;
        for (const di of unmatchedDets) {
          const det = detections[di];
          const cx = (det.bbox[0] + det.bbox[2]) / 2;
          const cy = (det.bbox[1] + det.bbox[3]) / 2;
          let d = Math.min(Math.hypot(trk.center[0] - cx, trk.center[1] - cy), Math.hypot(predX - cx, predY - cy));
          d *= trk.className === det.className ? 0.85 : 1.25;
          if (d < bestDist) {
            bestDist = d;
            bestIdx = di;
          }
        }
        if (bestIdx !== null) {
          matched.push([tid, bestIdx]);
          unmatchedTracks.delete(tid);
          unmatchedDets.delete(bestIdx);
        }
      }

      for (const [tid, di] of matched) {
        const det = detections[di];
        this.tracks.get(tid).update(det.bbox, det.confidence, det.className, fps);
      }

      for (const tid of unmatchedTracks) {
        const trk = this.tracks.get(tid);
        trk.markMissed();
        if (trk.missed > 20) this.tracks.delete(tid);
      }

      for (const di of unmatchedDets) {
        const det = detections[di];
        const nid = this.nextId++;
        this.totalUnique += 1;
        this.tracks.set(nid, new TrackState(nid, det.bbox, det.className, det.confidence, fps));
      }

      return this.tracks;
    }
  }

  class BrowserAccidentAnalyzer {
    constructor() {
      this.riskThreshold = 55;
      this.resetTemporal();
    }

    resetTemporal() {
      this.consecutiveAnomalyFrames = 0;
      this.smoothedRisk = 0;
      this.lastIncidentFrame = -999;
      this.activeIncident = null;
    }

    evaluatePair(v1, v2) {
      if (v1.className === "person" && v2.className === "person") return { risk: 0, reasons: [], ids: [] };

      const iou = calcIoU(v1.bbox, v2.bbox);
      const dx = v2.center[0] - v1.center[0];
      const dy = v2.center[1] - v1.center[1];
      const dist = Math.hypot(dx, dy);

      const relVx = v2.velocity[0] - v1.velocity[0];
      const relVy = v2.velocity[1] - v1.velocity[1];
      const closingSpeed = -(dx * relVx + dy * relVy) / (dist + 1e-5);

      const isPedVeh =
        (v1.className === "person" && v2.className !== "person") ||
        (v2.className === "person" && v1.className !== "person");
      if (isPedVeh) {
        const ped = v1.className === "person" ? v1 : v2;
        const veh = v1.className === "person" ? v2 : v1;
        const [px, py] = ped.center;
        const [vx1, vy1, vx2, vy2] = veh.bbox;
        if (
          (veh.className === "motorcycle" || veh.className === "bicycle") &&
          (iou > 0.05 || dist < 65 || (vx1 - 30 <= px && px <= vx2 + 30 && vy1 - 40 <= py && py <= vy2 + 40))
        ) {
          return { risk: 0, reasons: [], ids: [] };
        }
        const inside = vx1 - 25 <= px && px <= vx2 + 25 && vy1 - 25 <= py && py <= vy2 + 45;
        const rawH = Math.abs(veh.headingDeg - ped.headingDeg) % 360;
        const hDist = Math.min(rawH, 360 - rawH);
        if (inside && (hDist < 60 || Math.abs(veh.smoothedSpeed - ped.smoothedSpeed) < 3.5)) {
          return { risk: 0, reasons: [], ids: [] };
        }
      }

      const hDiff = Math.abs(v1.headingDeg - v2.headingDeg) % 360;
      const minH = Math.min(hDiff, 360 - hDiff);
      if (
        minH < 25 &&
        Math.abs(v1.smoothedSpeed - v2.smoothedSpeed) < 2 &&
        v1.smoothedDeceleration < 3.0 &&
        v2.smoothedDeceleration < 3.0
      ) {
        return { risk: 0, reasons: [], ids: [] };
      }

      const reasons = [];
      const maxPrior = Math.max(v1.maxRecentSpeed, v2.maxRecentSpeed);
      let scoreIou = 0;
      if (iou >= 0.12) {
        scoreIou = Math.min(100, 60 + (iou / 0.3) * 40);
        reasons.push(`Physical Collision Contact (IoU: ${iou.toFixed(2)})`);
      } else if (isPedVeh && (iou > 0.06 || (dist < 45 && maxPrior > 3.5))) {
        scoreIou = 90;
        reasons.push("Vehicle-Pedestrian Collision Contact");
      } else if (iou > 0.05) {
        scoreIou = (iou / 0.12) * 45;
        reasons.push(`Spatial Overlap (IoU: ${iou.toFixed(2)})`);
      } else if (dist < 95 && closingSpeed > 3.5) {
        scoreIou = Math.min(40, (closingSpeed / 10) * 40);
        reasons.push(`Converging Trajectory (${closingSpeed.toFixed(1)} px/f)`);
      }

      const maxDecel = Math.max(v1.smoothedDeceleration, v2.smoothedDeceleration);
      const nearOrContact = iou > 0.04 || (dist < 95 && closingSpeed > 2.8) || isPedVeh;
      let scoreDecel = 0;
      if (nearOrContact && maxDecel > 2.5) {
        scoreDecel = Math.min(100, (maxDecel / 10) * 100);
        reasons.push(`Sudden Deceleration (${maxDecel.toFixed(1)} px/f)`);
      }

      const maxAngle = Math.max(v1.headingDeltaDeg, v2.headingDeltaDeg);
      const maxSpeed = Math.max(v1.smoothedSpeed, v2.smoothedSpeed);
      let scoreAngle = 0;
      if (nearOrContact && maxAngle > 45 && maxSpeed > 2.0) {
        scoreAngle = Math.min(100, (maxAngle / 100) * 100);
        reasons.push(`Trajectory Deflection (${Math.round(maxAngle)}°)`);
      }

      const priorHigh = (v1.maxRecentSpeed > 3.8 && v1.isStopped) || (v2.maxRecentSpeed > 3.8 && v2.isStopped);
      let scoreStop = 0;
      if (priorHigh && (iou > 0.06 || scoreDecel > 18 || isPedVeh)) {
        scoreStop = 90;
        reasons.push("Post-Impact Stoppage");
      }

      if (scoreIou === 0 && scoreDecel === 0) return { risk: 0, reasons: [], ids: [] };

      const hasImpact = scoreDecel > 20 || scoreStop > 50 || scoreAngle > 30;
      let risk;
      if ((scoreIou >= 80 || iou >= 0.12) && hasImpact) {
        risk = Math.max(82, scoreIou * 0.35 + scoreDecel * 0.35 + scoreAngle * 0.15 + scoreStop * 0.15);
      } else {
        risk = scoreIou * 0.3 + scoreDecel * 0.3 + scoreAngle * 0.25 + scoreStop * 0.15;
      }

      risk = Math.min(100, Math.max(0, risk));
      return {
        risk,
        reasons,
        ids: risk >= 35 ? [v1.id, v2.id] : [],
      };
    }

    evaluateSingle(v) {
      if (!VEHICLE_CLASSES.has(v.className) || v.frameCount < 4) return { risk: 0, reasons: [] };
      const reasons = [];
      let risk = 0;
      if (v.maxRecentSpeed > 9.0 && v.smoothedSpeed < 1.5 && v.smoothedDeceleration > 7.0) {
        risk = 82;
        reasons.push(`Abrupt Impact Stop (${v.maxRecentSpeed.toFixed(1)} -> 0 px/f)`);
      }
      if (v.headingDeltaDeg > 75 && (v.smoothedSpeed > 3.0 || v.smoothedDeceleration > 5.0)) {
        risk = Math.max(risk, Math.min(92, 72 + (v.headingDeltaDeg / 180) * 20));
        reasons.push(`Violent Spin / Deflection (${Math.round(v.headingDeltaDeg)}°)`);
      }
      return { risk, reasons };
    }

    analyze(tracksMap, frameIdx) {
      const tracks = Array.from(tracksMap.values()).filter((t) => t.isConfirmed);
      let rawMax = 0;
      let primaryReason = "";
      let hazardIds = [];
      let involved = [];

      for (let i = 0; i < tracks.length; i++) {
        for (let j = i + 1; j < tracks.length; j++) {
          const res = this.evaluatePair(tracks[i], tracks[j]);
          if (res.risk > rawMax) {
            rawMax = res.risk;
            primaryReason = res.reasons.join(", ");
            hazardIds = res.ids;
            involved = [
              { id: tracks[i].id, class: tracks[i].className, speed: Number(tracks[i].smoothedSpeed.toFixed(1)) },
              { id: tracks[j].id, class: tracks[j].className, speed: Number(tracks[j].smoothedSpeed.toFixed(1)) },
            ];
          }
        }
      }

      for (const v of tracks) {
        const res = this.evaluateSingle(v);
        if (res.risk > rawMax) {
          rawMax = res.risk;
          primaryReason = res.reasons.join(", ");
          hazardIds = [v.id];
          involved = [{ id: v.id, class: v.className, speed: Number(v.smoothedSpeed.toFixed(1)) }];
        }
      }

      if (rawMax >= 35) {
        this.consecutiveAnomalyFrames += 1;
      } else {
        this.consecutiveAnomalyFrames = Math.max(0, this.consecutiveAnomalyFrames - 1);
      }

      if (rawMax >= this.riskThreshold || rawMax > this.smoothedRisk) {
        this.smoothedRisk = rawMax;
      } else {
        this.smoothedRisk = 0.85 * this.smoothedRisk + 0.15 * rawMax;
      }

      const displayRisk = Math.max(0, Math.min(100, this.smoothedRisk));
      let riskLevel = "NORMAL_TRAFFIC";
      let statusText = "Normal Traffic Flow";
      let isAlert = false;

      if (displayRisk >= 80 && this.consecutiveAnomalyFrames >= 3) {
        riskLevel = "CONFIRMED_ACCIDENT";
        statusText = "Confirmed Accident (Multi-Frame Validated)";
        isAlert = true;
      } else if (displayRisk >= 60) {
        riskLevel = "SUSPECTED_ACCIDENT";
        statusText = "Suspected Accident (Validating Temporal Persistence)";
        isAlert = true;
      } else if (displayRisk >= 35) {
        riskLevel = "COLLISION_RISK";
        statusText = "Potential Collision Risk";
        isAlert = displayRisk >= this.riskThreshold;
      }

      const shouldRecord = isAlert && displayRisk >= this.riskThreshold && frameIdx - this.lastIncidentFrame > 75;
      if (shouldRecord) {
        this.lastIncidentFrame = frameIdx;
      }

      return {
        isAlert,
        shouldRecord,
        riskScore: Number(displayRisk.toFixed(1)),
        riskLevel,
        statusText,
        reason: primaryReason || statusText,
        hazardIds,
        involved,
        consecutiveAnomalyFrames: this.consecutiveAnomalyFrames,
      };
    }
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve();
        return;
      }
      const s = document.createElement("script");
      s.src = src;
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(s);
    });
  }

  class BrowserAIEngine {
    constructor() {
      this.model = null;
      this.loadingPromise = null;
      this.tracker = new BrowserTracker();
      this.analyzer = new BrowserAccidentAnalyzer();
      this.running = false;
      this.rafId = null;
      this.busy = false;
      this.frameIdx = 0;
      this.confThreshold = 0.25;
      this.fps = 0;
      this.lastTick = 0;
      this.videoEl = null;
      this.canvasEl = null;
      this.onTelemetry = null;
      this.onIncident = null;
      this.onStatus = null;
    }

    async ensureModelLoaded() {
      if (this.model) return this.model;
      if (this.loadingPromise) return this.loadingPromise;

      this.loadingPromise = (async () => {
        if (this.onStatus) this.onStatus("Loading browser AI detection model...");
        await loadScript("/static/vendor/tfjs/tf.min.js");
        await loadScript("/static/vendor/tfjs/coco-ssd.min.js");
        const loaded = await global.cocoSsd.load({
          base: "lite_mobilenet_v2",
          modelUrl: "/static/models/coco-ssd/model.json",
        });
        this.model = loaded;
        if (this.onStatus) this.onStatus(null);
        return loaded;
      })();

      return this.loadingPromise;
    }

    attach(videoEl, canvasEl, { onTelemetry, onIncident, onStatus }) {
      this.videoEl = videoEl;
      this.canvasEl = canvasEl;
      this.onTelemetry = onTelemetry;
      this.onIncident = onIncident;
      this.onStatus = onStatus;
    }

    setConfidence(conf) {
      this.confThreshold = Math.max(0.1, Math.min(0.9, conf));
    }

    setRiskThreshold(t) {
      this.analyzer.riskThreshold = t;
    }

    resetState() {
      this.tracker.reset();
      this.analyzer.resetTemporal();
      this.frameIdx = 0;
      this.fps = 0;
      this.lastTick = 0;
      if (this.canvasEl) {
        const ctx = this.canvasEl.getContext("2d");
        ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);
      }
    }

    async start() {
      this.running = true;
      await this.ensureModelLoaded();
      if (!this.running) return;
      this.lastTick = performance.now();
      this._loop();
    }

    stop() {
      this.running = false;
      if (this.rafId) {
        cancelAnimationFrame(this.rafId);
        this.rafId = null;
      }
      if (this.canvasEl) {
        const ctx = this.canvasEl.getContext("2d");
        ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);
      }
    }

    _loop() {
      if (!this.running) return;
      this.rafId = requestAnimationFrame(() => this._step());
    }

    async _step() {
      if (!this.running || !this.videoEl || !this.canvasEl) return;

      if (
        !this.busy &&
        this.videoEl.readyState >= 2 &&
        !this.videoEl.paused &&
        !this.videoEl.ended &&
        this.videoEl.videoWidth > 0
      ) {
        this.busy = true;
        try {
          const vw = this.videoEl.videoWidth;
          const vh = this.videoEl.videoHeight;
          if (this.canvasEl.width !== vw || this.canvasEl.height !== vh) {
            this.canvasEl.width = vw;
            this.canvasEl.height = vh;
          }

          const t0 = performance.now();
          const preds = await this.model.detect(this.videoEl, 20, this.confThreshold);
          const infMs = performance.now() - t0;

          const now = performance.now();
          const dt = (now - this.lastTick) / 1000;
          this.lastTick = now;
          if (dt > 0) {
            const instFps = 1 / dt;
            this.fps = this.fps > 0 ? 0.85 * this.fps + 0.15 * instFps : instFps;
          }

          const detections = [];
          for (const p of preds) {
            if (!ROAD_USER_CLASSES.has(p.class) || p.score < this.confThreshold) continue;
            const [x, y, w, h] = p.bbox;
            detections.push({
              bbox: [x, y, x + w, y + h],
              className: p.class,
              confidence: p.score,
            });
          }

          this.frameIdx += 1;
          const tracksMap = this.tracker.update(detections, Math.max(5, this.fps || 15));
          const alertInfo = this.analyzer.analyze(tracksMap, this.frameIdx);

          this._drawOverlay(tracksMap, alertInfo);

          const confirmed = Array.from(tracksMap.values()).filter((t) => t.isConfirmed);
          const classCounts = {};
          const tracksSummary = confirmed.map((t) => {
            classCounts[t.className] = (classCounts[t.className] || 0) + 1;
            return {
              id: t.id,
              class: t.className,
              confidence: Number((t.confidence * 100).toFixed(1)),
              speed_display: t.getSpeedDisplay(),
              heading: Math.round(t.headingDeg),
              status: alertInfo.hazardIds.includes(t.id) ? "Hazard" : t.motionStatus,
            };
          });

          if (alertInfo.shouldRecord && this.onIncident) {
            const snapUrl = this._captureAnnotatedFrameDataUrl();
            this.onIncident({
              timestamp: new Date().toISOString().replace("T", " ").slice(0, 19),
              risk_score: alertInfo.riskScore,
              status: alertInfo.riskLevel,
              reason: alertInfo.reason,
              vehicles_involved: alertInfo.involved,
              vehicle_count: alertInfo.involved.length,
              snapshot_url: snapUrl,
            });
          }

          if (this.onTelemetry) {
            this.onTelemetry({
              fps: Number(this.fps.toFixed(1)),
              latency_ms: Number(infMs.toFixed(1)),
              active_vehicles: confirmed.length,
              total_detections: this.tracker.totalUnique,
              class_counts: classCounts,
              current_risk_score: alertInfo.riskScore,
              risk_level: alertInfo.riskLevel,
              status_text: alertInfo.statusText,
              consecutive_anomaly_frames: alertInfo.consecutiveAnomalyFrames,
              temporal_confirm_required: 3,
              alert: alertInfo.isAlert
                ? {
                    is_alert: true,
                    risk_level: alertInfo.riskLevel,
                    risk_score: alertInfo.riskScore,
                    reason: alertInfo.reason,
                  }
                : null,
              active_tracks: tracksSummary,
            });
          }
        } catch (err) {
          console.error("Browser AI frame error:", err);
        } finally {
          this.busy = false;
        }
      }

      this._loop();
    }

    _captureAnnotatedFrameDataUrl() {
      const snap = document.createElement("canvas");
      snap.width = this.canvasEl.width || this.videoEl.videoWidth;
      snap.height = this.canvasEl.height || this.videoEl.videoHeight;
      const ctx = snap.getContext("2d");
      ctx.drawImage(this.videoEl, 0, 0, snap.width, snap.height);
      ctx.drawImage(this.canvasEl, 0, 0, snap.width, snap.height);
      return snap.toDataURL("image/jpeg", 0.85);
    }

    _drawOverlay(tracksMap, alertInfo) {
      const ctx = this.canvasEl.getContext("2d");
      const w = this.canvasEl.width;
      const h = this.canvasEl.height;
      ctx.clearRect(0, 0, w, h);

      ctx.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";

      for (const trk of tracksMap.values()) {
        if (!trk.isConfirmed) continue;
        const isHazard = alertInfo.hazardIds.includes(trk.id) && alertInfo.riskLevel !== "NORMAL_TRAFFIC";
        let color = COLOR_LIME;
        if (isHazard) {
          color =
            alertInfo.riskLevel === "CONFIRMED_ACCIDENT" || alertInfo.riskLevel === "SUSPECTED_ACCIDENT"
              ? COLOR_RED
              : COLOR_AMBER;
        }

        // Subtle trajectory breadcrumb
        if (trk.history.length > 1) {
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.2;
          ctx.moveTo(trk.history[0][0], trk.history[0][1]);
          for (let i = 1; i < trk.history.length; i++) {
            ctx.lineTo(trk.history[i][0], trk.history[i][1]);
          }
          ctx.stroke();
        }

        // Thin bounding box
        const [x1, y1, x2, y2] = trk.bbox;
        ctx.strokeStyle = color;
        ctx.lineWidth = isHazard ? 2.5 : 1.5;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Compact label pill
        const label = `#${trk.id} ${trk.className.toUpperCase()} ${Math.round(trk.confidence * 100)}%`;
        const tw = ctx.measureText(label).width;
        const boxY = Math.max(20, y1);
        ctx.fillStyle = COLOR_CHARCOAL;
        ctx.fillRect(x1, boxY - 18, tw + 10, 18);
        ctx.fillStyle = color;
        ctx.fillText(label, x1 + 5, boxY - 5);
      }

      // Top minimal HUD bar
      ctx.fillStyle = "rgba(25, 29, 28, 0.82)";
      ctx.fillRect(0, 0, w, 26);
      ctx.fillStyle = COLOR_TEXT;
      ctx.fillText("AURA-VISION  |  BROWSER AI ENGINE", 10, 17);

      const badgeColor =
        alertInfo.riskLevel === "CONFIRMED_ACCIDENT" || alertInfo.riskLevel === "SUSPECTED_ACCIDENT"
          ? COLOR_RED
          : alertInfo.riskLevel === "COLLISION_RISK"
          ? COLOR_AMBER
          : COLOR_LIME;
      const rightTxt = `${alertInfo.riskLevel}  ${alertInfo.riskScore}%`;
      const rw = ctx.measureText(rightTxt).width;
      ctx.fillStyle = badgeColor;
      ctx.fillText(rightTxt, w - rw - 10, 17);
    }
  }

  global.AuraBrowserAI = new BrowserAIEngine();
})(window);
