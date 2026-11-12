/**
 * AURA-VISION Frontend Controller
 * Manages view navigation, live telemetry polling, video source switching,
 * model & inference configuration, 4-tier accident alerts, incident exports,
 * and native modal inspection.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation & Views
  const navTabs = document.querySelectorAll(".nav-tab");
  const views = {
    overview: document.getElementById("view-overview"),
    monitoring: document.getElementById("view-monitoring"),
    incidents: document.getElementById("view-incidents"),
  };
  const ctaLaunch = document.getElementById("cta-launch");
  const ctaExplore = document.getElementById("cta-explore");
  const techOverview = document.getElementById("tech-overview");

  // Header Status
  const headerModelChip = document.getElementById("header-model-chip");
  const systemStatePill = document.getElementById("system-state-pill");
  const systemStateText = document.getElementById("system-state-text");
  const navIncidentCount = document.getElementById("nav-incident-count");

  // Banners
  const errorBanner = document.getElementById("error-banner");
  const errorBannerText = document.getElementById("error-banner-text");
  const accidentBanner = document.getElementById("accident-banner");
  const bannerTag = document.getElementById("banner-tag");
  const bannerTitle = document.getElementById("banner-title");
  const bannerReason = document.getElementById("banner-reason");
  const bannerRisk = document.getElementById("banner-risk");

  // Video Monitor
  const videoStream = document.getElementById("video-stream");
  const monitorLiveDot = document.getElementById("monitor-live-dot");
  const activeSourceTitle = document.getElementById("active-source-title");
  const monitorResMeta = document.getElementById("monitor-res-meta");

  // Analytics & Accident Monitoring
  const valFps = document.getElementById("val-fps");
  const valLatency = document.getElementById("val-latency");
  const valActiveVehicles = document.getElementById("val-active-vehicles");
  const valTotalVehicles = document.getElementById("val-total-vehicles");
  const deviceNameEl = document.getElementById("device-name");
  const vramDisplay = document.getElementById("vram-display");
  const classBreakdownChips = document.getElementById("class-breakdown-chips");
  const supportedClassesNote = document.getElementById("supported-classes-note");

  const valRiskScore = document.getElementById("val-risk-score");
  const valRiskBadge = document.getElementById("val-risk-badge");
  const riskGaugeBar = document.getElementById("risk-gauge-bar");
  const valRiskSubtext = document.getElementById("val-risk-subtext");
  const valTemporalStatus = document.getElementById("val-temporal-status");
  const latestIncidentTime = document.getElementById("latest-incident-time");
  const latestIncidentSummary = document.getElementById("latest-incident-summary");
  const btnViewLatestIncident = document.getElementById("btn-view-latest-incident");

  // Controls
  const btnStart = document.getElementById("btn-start");
  const btnStop = document.getElementById("btn-stop");
  const btnSnapshot = document.getElementById("btn-snapshot");
  const srcWebcam = document.getElementById("src-webcam");
  const srcUploadTrigger = document.getElementById("src-upload-trigger");
  const videoFileInput = document.getElementById("video-file-input");
  const srcSampleNormal = document.getElementById("src-sample-normal");
  const srcSampleAccident = document.getElementById("src-sample-accident");
  const sourceButtons = [srcWebcam, srcUploadTrigger, srcSampleNormal, srcSampleAccident];

  const modelSelector = document.getElementById("model-selector");
  const imgszSelector = document.getElementById("imgsz-selector");
  const confSlider = document.getElementById("conf-slider");
  const confLabel = document.getElementById("conf-label");
  const sensitivitySlider = document.getElementById("sensitivity-slider");
  const sensitivityLabel = document.getElementById("sensitivity-label");
  const audioAlarmToggle = document.getElementById("audio-alarm-toggle");
  const btnRunBenchmark = document.getElementById("btn-run-benchmark");
  const benchmarkResults = document.getElementById("benchmark-results");

  // Tables & Incident Actions
  const telemetryTbody = document.getElementById("telemetry-tbody");
  const activeTrackCount = document.getElementById("active-track-count");
  const historyTbody = document.getElementById("history-tbody");
  const valAccidentsCount = document.getElementById("val-accidents-count");
  const btnExportJson = document.getElementById("btn-export-json");
  const btnExportCsv = document.getElementById("btn-export-csv");
  const btnClearHistory = document.getElementById("btn-clear-history");

  // Native Dialog Elements
  const snapshotDialog = document.getElementById("snapshot-dialog");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalImage = document.getElementById("modal-image");
  const modalTitle = document.getElementById("modal-title");
  const modalCaption = document.getElementById("modal-caption");
  const modalDownloadBtn = document.getElementById("modal-download-btn");

  let lastAlertTimestamp = 0;
  let audioContext = null;
  let latestIncidentCache = null;
  let staticDemoMode = false;
  let cachedEvents = [];
  let telemetryTimer = null;
  let eventsTimer = null;
  let demoVideoEl = null;

  function enableStaticDemoMode() {
    if (staticDemoMode) return;
    staticDemoMode = true;
    if (telemetryTimer) clearInterval(telemetryTimer);
    if (eventsTimer) clearInterval(eventsTimer);

    headerModelChip.textContent = "YOLOv8n · Cloud Preview";
    systemStatePill.className = "state-badge state-standby";
    systemStateText.textContent = "Cloud Preview";
    deviceNameEl.textContent = "Static CDN (Run app.py locally for live inference)";
    monitorResMeta.textContent = "Sample Video · Static Mode";

    // Swap MJPEG <img> for HTML5 <video> playing local sample clips
    if (videoStream && videoStream.parentElement) {
      videoStream.hidden = true;
      demoVideoEl = document.createElement("video");
      demoVideoEl.id = "demo-video-player";
      demoVideoEl.className = "video-img";
      demoVideoEl.controls = true;
      demoVideoEl.autoplay = true;
      demoVideoEl.loop = true;
      demoVideoEl.muted = true;
      demoVideoEl.playsInline = true;
      demoVideoEl.src = "/sample_data/accident_scenario.mp4";
      videoStream.parentElement.appendChild(demoVideoEl);
    }

    showError(
      "Cloud Preview Mode: Sample videos, incident snapshots, and JSON/CSV exports are active. Run 'python app.py' locally for live YOLOv8n + ByteTrack + webcam inference."
    );
    fetchStaticEventsLog();
  }

  async function fetchStaticEventsLog() {
    try {
      const res = await fetch("/accident_events/events_log.json");
      if (!res.ok) return;
      const events = await res.json();
      cachedEvents = Array.isArray(events) ? events : [];
      navIncidentCount.textContent = String(cachedEvents.length);
      valAccidentsCount.textContent = `${cachedEvents.length} ${cachedEvents.length === 1 ? "event" : "events"}`;
      if (cachedEvents.length > 0) {
        latestIncidentCache = cachedEvents[0];
        latestIncidentTime.textContent = latestIncidentCache.timestamp || "";
        latestIncidentSummary.textContent = `#${String(latestIncidentCache.event_id).padStart(4, "0")} · ${latestIncidentCache.status} (${latestIncidentCache.risk_score}%) — ${latestIncidentCache.reason}`;
        btnViewLatestIncident.hidden = false;
      }
      renderEventsLog(cachedEvents);
    } catch (err) {
      console.error("Static events load error:", err);
    }
  }

  // --- View Switching ---
  function switchView(targetView) {
    Object.entries(views).forEach(([key, section]) => {
      if (section) {
        section.classList.toggle("is-active", key === targetView);
      }
    });
    navTabs.forEach((tab) => {
      tab.classList.toggle("is-active", tab.dataset.view === targetView);
    });
    if (targetView === "incidents") {
      fetchEventsLog();
    }
  }

  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
  });

  if (ctaLaunch) {
    ctaLaunch.addEventListener("click", () => {
      switchView("monitoring");
    });
  }

  if (ctaExplore && techOverview) {
    ctaExplore.addEventListener("click", () => {
      techOverview.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  // --- Error Helpers ---
  function showError(message) {
    if (!message) {
      clearError();
      return;
    }
    errorBannerText.textContent = message;
    errorBanner.hidden = false;
  }

  function clearError() {
    errorBannerText.textContent = "";
    errorBanner.hidden = true;
  }

  // --- Native Accessible Dialog ---
  function openSnapshotDialog(url, title, reason) {
    modalImage.src = url;
    modalTitle.textContent = title;
    modalCaption.textContent = reason || "Incident evidence snapshot";
    modalDownloadBtn.href = url;
    if (typeof snapshotDialog.showModal === "function") {
      snapshotDialog.showModal();
    } else {
      window.open(url, "_blank", "noopener");
    }
  }

  modalCloseBtn.addEventListener("click", () => snapshotDialog.close());
  snapshotDialog.addEventListener("click", (e) => {
    if (e.target === snapshotDialog) {
      snapshotDialog.close();
    }
  });

  // --- Subtle Optional Audio Chime ---
  function playAlertBeep(isConfirmed = false) {
    if (!audioAlarmToggle || !audioAlarmToggle.checked) return;

    const now = Date.now();
    if (now - lastAlertTimestamp < (isConfirmed ? 2500 : 4000)) return;
    lastAlertTimestamp = now;

    try {
      if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContext.state === "suspended") {
        audioContext.resume();
      }
      const osc = audioContext.createOscillator();
      const gain = audioContext.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(isConfirmed ? 840 : 660, audioContext.currentTime);
      gain.gain.setValueAtTime(0.14, audioContext.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.005, audioContext.currentTime + 0.25);
      osc.connect(gain);
      gain.connect(audioContext.destination);
      osc.start();
      osc.stop(audioContext.currentTime + 0.26);
    } catch (_) {
      // Ignore audio errors if blocked by browser autoplay policy
    }
  }

  // --- Source Labels ---
  const SOURCE_TITLES = {
    webcam: "Source: Live Webcam",
    sample_normal: "Source: Sample Traffic",
    sample_accident: "Source: Sample Accident",
    upload: "Source: Uploaded Video",
  };

  function syncSourceButtons(source) {
    const map = {
      webcam: srcWebcam,
      upload: srcUploadTrigger,
      sample_normal: srcSampleNormal,
      sample_accident: srcSampleAccident,
    };
    sourceButtons.forEach((b) => b.classList.remove("is-active"));
    if (map[source]) {
      map[source].classList.add("is-active");
    }
  }

  // --- Telemetry Polling ---
  async function fetchTelemetry() {
    if (staticDemoMode) return;
    try {
      const res = await fetch("/api/stats");
      if (res.status === 404) {
        enableStaticDemoMode();
        return;
      }
      if (!res.ok) return;
      const data = await res.json();

      // Surface backend errors cleanly
      if (data.error) {
        showError(data.error);
      }

      const cleanModel = (data.model_name || "yolov8n.pt").replace(".pt", "").toUpperCase();
      const devLabel = data.device_code && data.device_code.startsWith("cuda") ? "CUDA" : "CPU";
      headerModelChip.textContent = `${cleanModel} · ${devLabel}`;
      monitorResMeta.textContent = `${data.imgsz || 640} px · ${cleanModel}`;
      deviceNameEl.textContent = data.device || devLabel;

      if (data.vram && data.vram.has_nvml) {
        vramDisplay.textContent = `VRAM ${data.vram.used_mb}/${data.vram.total_mb} MB`;
      } else {
        vramDisplay.textContent = "";
      }

      if (data.is_running) {
        systemStatePill.className = "state-badge state-live";
        systemStateText.textContent = "Live";
        monitorLiveDot.classList.add("is-live");
      } else {
        systemStatePill.className = "state-badge state-standby";
        systemStateText.textContent = "Standby";
        monitorLiveDot.classList.remove("is-live");
      }

      if (data.source && SOURCE_TITLES[data.source] && data.source !== "upload") {
        activeSourceTitle.textContent = SOURCE_TITLES[data.source];
        syncSourceButtons(data.source);
      }

      // Analytics numbers
      valFps.textContent = (data.fps || 0).toFixed(1);
      valLatency.textContent = `${(data.latency_ms || 0).toFixed(1)} ms`;
      valActiveVehicles.textContent = String(data.active_vehicles ?? 0);
      valTotalVehicles.textContent = String(data.total_detections ?? 0);

      // Incident counts
      const totalIncidents = data.accidents_count ?? 0;
      navIncidentCount.textContent = String(totalIncidents);
      valAccidentsCount.textContent = `${totalIncidents} ${totalIncidents === 1 ? "event" : "events"}`;

      // Class breakdown chips
      renderClassBreakdown(data.class_counts || {});
      if (Array.isArray(data.supported_classes) && data.supported_classes.length > 0) {
        const clsStr = data.supported_classes.map((c) => c.replace("_", " ")).join(", ");
        supportedClassesNote.textContent = `Active model classes: ${clsStr}. Custom weights supported for scooter & three-wheeler.`;
      }

      // 4-Tier Accident Monitoring
      const risk = data.current_risk_score || 0;
      const tier = data.risk_level || "NORMAL_TRAFFIC";
      valRiskScore.textContent = `${risk.toFixed(1)}%`;
      riskGaugeBar.style.width = `${Math.min(100, Math.max(0, risk))}%`;
      valRiskSubtext.textContent = data.status_text || "Normal Traffic Flow";

      const consec = data.consecutive_anomaly_frames || 0;
      const reqFrames = data.temporal_confirm_required || 3;
      valTemporalStatus.textContent = `${consec} / ${reqFrames} frames`;

      valRiskBadge.textContent = tier;
      valRiskBadge.className = "tier-badge";
      riskGaugeBar.className = "progress-fill";

      if (tier === "CONFIRMED_ACCIDENT") {
        valRiskBadge.classList.add("tier-confirmed");
        riskGaugeBar.classList.add("is-red");
      } else if (tier === "SUSPECTED_ACCIDENT") {
        valRiskBadge.classList.add("tier-suspected");
        riskGaugeBar.classList.add("is-red");
      } else if (tier === "COLLISION_RISK") {
        valRiskBadge.classList.add("tier-risk");
        riskGaugeBar.classList.add("is-amber");
      } else {
        valRiskBadge.classList.add("tier-normal");
      }

      // Alert Banner
      if (data.alert && data.alert.is_alert) {
        const alt = data.alert;
        const isCritical = alt.risk_level === "CONFIRMED_ACCIDENT" || alt.risk_level === "SUSPECTED_ACCIDENT";
        accidentBanner.className = `status-banner ${isCritical ? "is-critical" : "is-warning"}`;
        bannerTag.textContent = alt.risk_level;
        bannerTitle.textContent =
          alt.risk_level === "CONFIRMED_ACCIDENT"
            ? "Confirmed Accident Detected (Multi-Frame Validated)"
            : alt.risk_level === "SUSPECTED_ACCIDENT"
            ? "Suspected Accident — Validating Temporal Persistence"
            : "Potential Collision Risk";
        bannerReason.textContent = alt.reason || "Collision kinematics anomaly";
        bannerRisk.textContent = `${alt.risk_score}%`;
        if (isCritical) {
          playAlertBeep(alt.risk_level === "CONFIRMED_ACCIDENT");
        }
      } else {
        accidentBanner.className = "status-banner is-hidden";
      }

      // Latest Incident Summary
      latestIncidentCache = data.latest_incident || null;
      if (latestIncidentCache) {
        latestIncidentTime.textContent = latestIncidentCache.timestamp || "";
        latestIncidentSummary.textContent = `#${String(latestIncidentCache.event_id).padStart(4, "0")} · ${latestIncidentCache.status} (${latestIncidentCache.risk_score}%) — ${latestIncidentCache.reason}`;
        btnViewLatestIncident.hidden = false;
      } else {
        latestIncidentTime.textContent = "None";
        latestIncidentSummary.textContent = "No collision incidents recorded.";
        btnViewLatestIncident.hidden = true;
      }

      // Active Tracks Table
      renderActiveTracks(data.active_tracks || []);
    } catch (err) {
      console.error("Telemetry polling error:", err);
    }
  }

  function renderClassBreakdown(counts) {
    classBreakdownChips.innerHTML = "";
    const entries = Object.entries(counts);
    if (entries.length === 0) {
      const empty = document.createElement("span");
      empty.className = "class-chip";
      empty.textContent = "No active detections";
      classBreakdownChips.appendChild(empty);
      return;
    }
    entries.sort((a, b) => b[1] - a[1]);
    entries.forEach(([cls, count]) => {
      const chip = document.createElement("span");
      chip.className = "class-chip";
      const label = cls.replace("_", " ");
      chip.appendChild(document.createTextNode(`${label}: `));
      const strong = document.createElement("strong");
      strong.textContent = String(count);
      chip.appendChild(strong);
      classBreakdownChips.appendChild(chip);
    });
  }

  function renderActiveTracks(tracks) {
    activeTrackCount.textContent = `${tracks.length} active`;
    telemetryTbody.innerHTML = "";

    if (!tracks || tracks.length === 0) {
      const tr = document.createElement("tr");
      tr.className = "empty-row";
      const td = document.createElement("td");
      td.colSpan = 6;
      td.textContent = "No active road users currently tracked.";
      tr.appendChild(td);
      telemetryTbody.appendChild(tr);
      return;
    }

    tracks.forEach((t) => {
      const tr = document.createElement("tr");
      const isHazard = t.status === "Hazard";
      if (isHazard) tr.className = "row-hazard";

      const cells = [
        { text: `#${t.id}`, cls: "tabular" },
        { text: String(t.class || "").replace("_", " ").toUpperCase(), cls: "" },
        { text: t.confidence != null ? `${t.confidence}%` : "—", cls: "tabular" },
        { text: t.speed_display || `${t.speed} px/s`, cls: "tabular" },
        { text: `${t.heading}°`, cls: "tabular" },
        { text: t.status || "Cruising", cls: "" },
      ];

      cells.forEach((c) => {
        const td = document.createElement("td");
        if (c.cls) td.className = c.cls;
        td.textContent = c.text;
        tr.appendChild(td);
      });

      telemetryTbody.appendChild(tr);
    });
  }

  // --- Incident History ---
  async function fetchEventsLog() {
    if (staticDemoMode) {
      await fetchStaticEventsLog();
      return;
    }
    try {
      const res = await fetch("/api/events");
      if (res.status === 404) {
        enableStaticDemoMode();
        return;
      }
      if (!res.ok) return;
      const data = await res.json();
      cachedEvents = data.events || [];
      renderEventsLog(cachedEvents);
    } catch (err) {
      console.error("Events log error:", err);
    }
  }

  function getTierBadgeClass(status) {
    const s = String(status || "").toUpperCase();
    if (s.includes("CONFIRMED") || s.includes("POSSIBLE")) return "tier-confirmed";
    if (s.includes("SUSPECTED")) return "tier-suspected";
    if (s.includes("RISK") || s.includes("CAUTION")) return "tier-risk";
    return "tier-normal";
  }

  function renderEventsLog(events) {
    historyTbody.innerHTML = "";
    if (!events || events.length === 0) {
      const tr = document.createElement("tr");
      tr.className = "empty-row";
      const td = document.createElement("td");
      td.colSpan = 7;
      td.textContent = "No incidents recorded yet.";
      tr.appendChild(td);
      historyTbody.appendChild(tr);
      return;
    }

    events.forEach((e) => {
      const tr = document.createElement("tr");
      const idStr = `#${String(e.event_id).padStart(4, "0")}`;

      const tdId = document.createElement("td");
      tdId.className = "tabular";
      tdId.textContent = idStr;

      const tdTime = document.createElement("td");
      tdTime.className = "tabular";
      tdTime.textContent = e.timestamp || "";

      const tdThumb = document.createElement("td");
      const thumbBtn = document.createElement("button");
      thumbBtn.type = "button";
      thumbBtn.className = "thumb-btn";
      thumbBtn.title = `Inspect ${idStr} snapshot`;
      const img = document.createElement("img");
      img.src = e.snapshot_url;
      img.alt = `Incident ${idStr} snapshot`;
      img.className = "table-thumb";
      img.loading = "lazy";
      thumbBtn.appendChild(img);
      thumbBtn.addEventListener("click", () => {
        openSnapshotDialog(
          e.snapshot_url,
          `Incident ${idStr} · ${e.risk_score}%`,
          `${e.status} — ${e.reason}`
        );
      });
      tdThumb.appendChild(thumbBtn);

      const tdStatus = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = `tier-badge ${getTierBadgeClass(e.status)}`;
      badge.textContent = e.status || "SUSPECTED_ACCIDENT";
      tdStatus.appendChild(badge);

      const tdRisk = document.createElement("td");
      tdRisk.className = "tabular";
      tdRisk.textContent = `${e.risk_score}%`;

      const tdVeh = document.createElement("td");
      const vehSummary = (e.vehicles_involved || [])
        .map((v) => `#${v.id} ${v.class}`)
        .join(", ");
      tdVeh.textContent = `${e.vehicle_count ?? (e.vehicles_involved || []).length} (${vehSummary || "—"})`;

      const tdReason = document.createElement("td");
      tdReason.textContent = e.reason || "";

      tr.append(tdId, tdTime, tdThumb, tdStatus, tdRisk, tdVeh, tdReason);
      historyTbody.appendChild(tr);
    });
  }

  if (btnViewLatestIncident) {
    btnViewLatestIncident.addEventListener("click", () => {
      if (!latestIncidentCache) return;
      openSnapshotDialog(
        latestIncidentCache.snapshot_url,
        `Incident #${String(latestIncidentCache.event_id).padStart(4, "0")} · ${latestIncidentCache.risk_score}%`,
        `${latestIncidentCache.status} — ${latestIncidentCache.reason}`
      );
    });
  }

  // --- Stream Controls ---
  async function postJson(url, payload = {}) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.success === false) {
      throw new Error(data.error || `Request failed (${res.status})`);
    }
    return data;
  }

  btnStart.addEventListener("click", async () => {
    clearError();
    try {
      await postJson("/api/start_detection");
    } catch (err) {
      showError(err.message);
    }
  });

  btnStop.addEventListener("click", async () => {
    clearError();
    try {
      await postJson("/api/stop_detection");
    } catch (err) {
      showError(err.message);
    }
  });

  btnSnapshot.addEventListener("click", async () => {
    clearError();
    try {
      btnSnapshot.disabled = true;
      const data = await postJson("/api/capture_snapshot");
      openSnapshotDialog(data.snapshot_url, "Manual Snapshot Capture", "Captured from active video stream.");
    } catch (err) {
      showError(err.message);
    } finally {
      btnSnapshot.disabled = false;
    }
  });

  async function switchSource(source, btn, titleText) {
    clearError();
    syncSourceButtons(source);
    activeSourceTitle.textContent = titleText;
    if (staticDemoMode) {
      if (source === "sample_normal" && demoVideoEl) {
        demoVideoEl.src = "/sample_data/normal_traffic.mp4";
      } else if (source === "sample_accident" && demoVideoEl) {
        demoVideoEl.src = "/sample_data/accident_scenario.mp4";
      } else {
        showError("Live webcam and video uploads require running the local Flask server ('python app.py').");
      }
      return;
    }
    try {
      await postJson("/api/switch_source", { source });
    } catch (err) {
      showError(err.message);
    }
  }

  srcWebcam.addEventListener("click", () => switchSource("webcam", srcWebcam, "Source: Live Webcam"));
  srcSampleNormal.addEventListener("click", () => switchSource("sample_normal", srcSampleNormal, "Source: Sample Traffic"));
  srcSampleAccident.addEventListener("click", () => switchSource("sample_accident", srcSampleAccident, "Source: Sample Accident"));

  srcUploadTrigger.addEventListener("click", () => {
    videoFileInput.value = "";
    videoFileInput.click();
  });

  videoFileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    clearError();
    syncSourceButtons("upload");
    activeSourceTitle.textContent = `Uploading: ${file.name}...`;

    const formData = new FormData();
    formData.append("video", file);

    try {
      const res = await fetch("/api/upload_video", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || "Video upload failed.");
      }
      activeSourceTitle.textContent = `Source: ${file.name}`;
    } catch (err) {
      showError(err.message);
      activeSourceTitle.textContent = "Source: Upload Failed";
    }
  });

  // --- Engine Settings ---
  modelSelector.addEventListener("change", async () => {
    clearError();
    const chosen = modelSelector.value;
    modelSelector.disabled = true;
    try {
      await postJson("/api/set_model", { model_name: chosen });
    } catch (err) {
      showError(err.message);
    } finally {
      modelSelector.disabled = false;
    }
  });

  imgszSelector.addEventListener("change", async () => {
    clearError();
    try {
      await postJson("/api/set_inference_params", { imgsz: parseInt(imgszSelector.value, 10) });
    } catch (err) {
      showError(err.message);
    }
  });

  confSlider.addEventListener("input", (e) => {
    confLabel.textContent = Number(e.target.value).toFixed(2);
  });

  confSlider.addEventListener("change", async (e) => {
    clearError();
    try {
      await postJson("/api/set_inference_params", { conf: parseFloat(e.target.value) });
    } catch (err) {
      showError(err.message);
    }
  });

  sensitivitySlider.addEventListener("input", (e) => {
    sensitivityLabel.textContent = `${e.target.value}%`;
  });

  sensitivitySlider.addEventListener("change", async (e) => {
    clearError();
    try {
      await postJson("/api/set_threshold", { threshold: parseFloat(e.target.value) });
    } catch (err) {
      showError(err.message);
    }
  });

  btnRunBenchmark.addEventListener("click", async () => {
    clearError();
    btnRunBenchmark.disabled = true;
    btnRunBenchmark.textContent = "Benchmarking...";
    benchmarkResults.hidden = false;
    benchmarkResults.textContent = "Running timed inference across 384px, 480px, 640px...";

    try {
      const data = await postJson("/api/benchmark", { model_name: modelSelector.value });
      benchmarkResults.innerHTML = "";

      const head = document.createElement("div");
      head.textContent = `${data.model.toUpperCase()} (${data.device})`;
      benchmarkResults.appendChild(head);

      const grid = document.createElement("div");
      grid.className = "bench-grid";
      Object.entries(data.benchmarks || {}).forEach(([k, val]) => {
        const sz = k.replace("imgsz_", "");
        const item = document.createElement("div");
        item.className = "bench-item";

        const sEl = document.createElement("span");
        sEl.textContent = `${sz} px`;
        const lEl = document.createElement("strong");
        lEl.className = "tabular";
        lEl.textContent = `${val.latency_ms} ms`;
        const fEl = document.createElement("span");
        fEl.className = "tabular";
        fEl.textContent = `${val.fps} FPS`;

        item.append(sEl, lEl, fEl);
        grid.appendChild(item);
      });
      benchmarkResults.appendChild(grid);
    } catch (err) {
      benchmarkResults.textContent = `Benchmark failed: ${err.message}`;
    } finally {
      btnRunBenchmark.disabled = false;
      btnRunBenchmark.textContent = "Run Inference Benchmark";
    }
  });

  function triggerClientDownload(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // --- Exports & History Management ---
  btnExportJson.addEventListener("click", () => {
    if (staticDemoMode) {
      triggerClientDownload(JSON.stringify(cachedEvents, null, 2), "accident_events_export.json", "application/json");
      return;
    }
    window.location.href = "/api/export_events?format=json";
  });

  btnExportCsv.addEventListener("click", () => {
    if (staticDemoMode) {
      const rows = [["Event ID", "Timestamp", "Risk Score (%)", "Status", "Reason", "Vehicles Count", "Snapshot URL"]];
      cachedEvents.forEach((e) => {
        rows.push([
          e.event_id,
          e.timestamp,
          e.risk_score,
          e.status,
          `"${String(e.reason || "").replace(/"/g, '""')}"`,
          e.vehicle_count ?? (e.vehicles_involved || []).length,
          e.snapshot_url,
        ]);
      });
      triggerClientDownload(rows.map((r) => r.join(",")).join("\n"), "accident_events_export.csv", "text/csv");
      return;
    }
    window.location.href = "/api/export_events?format=csv";
  });

  btnClearHistory.addEventListener("click", async () => {
    if (staticDemoMode) {
      showError("Clearing server history requires running the local Flask server.");
      return;
    }
    if (!confirm("Clear all recorded incident logs and snapshots?")) return;
    clearError();
    try {
      await postJson("/api/clear_history");
      fetchEventsLog();
      fetchTelemetry();
    } catch (err) {
      showError(err.message);
    }
  });

  // --- Start Loops ---
  fetchTelemetry();
  fetchEventsLog();
  telemetryTimer = setInterval(fetchTelemetry, 400);
  eventsTimer = setInterval(fetchEventsLog, 3000);
});
