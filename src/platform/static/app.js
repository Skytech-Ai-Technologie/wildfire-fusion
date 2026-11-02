/* Live platform UI logic: poll /api/status, render telemetry, post controls. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var els = {
    conn: $("pill-conn"), state: $("state-badge"), stateText: $("state-text"),
    latency: $("latency-chip"), source: $("chip-source"), modelInfo: $("model-info"),
    massF: $("mass-f"), barMass: $("bar-mass"), barTau: $("bar-tau"), tauLabel: $("tau-label"),
    mF: $("m-F"), mNotF: $("m-notF"), mTheta: $("m-Theta"), mDiv: $("m-div"),
    modalities: $("modalities"), policy: $("chip-policy"), toggle: $("btn-toggle"),
    tau: $("tau"), tauVal: $("tau-val"), conf: $("conf"), confVal: $("conf-val"),
    policySel: $("policy"), video: $("video"),
    srcCamera: $("src-camera"), srcFile: $("src-file"), srcFilePath: $("src-file-path"),
    srcYoutube: $("src-youtube"), cameraNote: $("camera-note"), fileNote: $("file-note"),
    youtubeNote: $("youtube-note"), fileInput: $("file-input"),
    btnUpload: $("btn-upload"), uploadNote: $("upload-note"), dropHint: $("drop-hint")
  };

  var lastState = null;

  function fmt(v, d) {
    return (v === null || v === undefined || isNaN(v)) ? "—" : Number(v).toFixed(d === undefined ? 3 : d);
  }

  function setConn(ok, text) {
    els.conn.textContent = text;
    els.conn.className = "pill " + (ok ? "ok" : "bad");
  }

  function renderModalities(mods) {
    if (!mods || !mods.length) {
      els.modalities.innerHTML = '<p class="muted">no frame processed yet</p>';
      return;
    }
    var html = "";
    mods.forEach(function (m) {
      var raw = Number(m.raw_pi || 0);
      var ref = m.refined_pi === null || m.refined_pi === undefined ? null : Number(m.refined_pi);
      html +=
        '<div class="mod' + (m.simulated ? ' sim' : '') + '">' +
          '<div class="mod-head">' +
            '<span class="mod-name">' + (m.name === "c" ? "C · " : m.name === "d" ? "D · " : "S · ") +
              m.label + '</span>' +
            '<span class="mod-tag' + (m.simulated ? ' sim' : '') + '">' +
              (m.simulated ? "simulated" : (m.source || "")) + '</span>' +
          '</div>' +
          '<div class="mod-bars">' +
            '<span>raw π</span><span class="mini"><i style="width:' +
              Math.min(100, raw * 100).toFixed(1) + '%"></i></span>' +
              '<span class="v">' + raw.toFixed(3) + '</span>' +
            '<span>refined π̂</span><span class="mini ref"><i style="width:' +
              Math.min(100, (ref === null ? 0 : ref) * 100).toFixed(1) + '%"></i></span>' +
              '<span class="v">' + (ref === null ? "—" : ref.toFixed(3)) + '</span>' +
          '</div>' +
          (m.readings ? '<div class="mod-bars"><span>readings</span>' +
            '<span class="v" style="grid-column:2/4;text-align:right;font-size:.72rem">' +
            Object.keys(m.readings).map(function (k) { return k + "=" + m.readings[k]; }).join("  ") +
            '</span></div>' : '') +
        '</div>';
    });
    els.modalities.innerHTML = html;
  }

  function apply(s) {
    lastState = s;

    if (s.error) {
      var box = els.modalities.querySelector(".error");
      if (!box) {
        box = document.createElement("div");
        box.className = "error";
        els.modalities.parentNode.insertBefore(box, els.modalities);
      }
      box.textContent = "Capture/inference error: " + s.error;
    }

    var st = s.state || "idle";
    var alarmed = st === "alarm" || st === "alarm_confirmed";
    els.state.className = "state-badge" + (alarmed ? (st === "alarm" ? " alarm" : " confirmed") : "");
    els.stateText.textContent = (st === "alarm_confirmed" ? "ALARM CONFIRMED" : st.toUpperCase());

    els.latency.textContent = (s.fps !== undefined ? s.fps.toFixed(1) + " fps · " : "") +
      (s.latency_ms !== undefined ? s.latency_ms + " ms" : "—");
    var srcLabel = s.source || "—";
    if (s.source_status) { srcLabel += " · " + s.source_status; }
    els.source.textContent = srcLabel;
    els.modelInfo.textContent =
      "model: " + (s.arch || "?") + "  ·  device: " + (s.device || "?") +
      "  ·  LWIR: " + (s.lwir_source || "—") +
      "  ·  window: " + ((s.window && s.window.filled) || 0) + "/" + ((s.window && s.window.T) || 0) +
      " frames" + (s.contextual_loaded === false ? "  ·  identity refinement" : "");

    var tau = Number(s.tau !== undefined ? s.tau : 0.78);
    var f = Number((s.fused_mass && s.fused_mass.F) || 0);
    els.massF.textContent = fmt(f);
    els.mF.textContent = fmt(s.fused_mass && s.fused_mass.F);
    els.mNotF.textContent = fmt(s.fused_mass && s.fused_mass.not_F);
    els.mTheta.textContent = fmt(s.fused_mass && s.fused_mass["Θ"]);
    els.mDiv.textContent = (s.divergence && s.divergence.length)
      ? s.divergence.map(function (d) { return Number(d).toFixed(3); }).join(" / ") : "—";

    els.barMass.style.width = Math.min(100, f * 100).toFixed(1) + "%";
    els.barMass.className = "bar-fill" + (alarmed ? " alarm" : "");
    els.barTau.style.left = (tau * 100).toFixed(1) + "%";
    els.tauLabel.textContent = tau.toFixed(2);

    if (document.activeElement !== els.tau) {
      els.tau.value = tau;
      els.tauVal.textContent = tau.toFixed(2);
    }
    els.policy.textContent = els.policySel.value;
    renderModalities(s.modalities);

    els.toggle.textContent = s.running ? "Stop" : "Start";
  }

  function poll() {
    fetch("/api/status", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (s) { setConn(true, "live"); apply(s); })
      .catch(function () { setConn(false, "disconnected"); });
  }

  function post(cmd) {
    return fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cmd)
    }).then(function (r) { return r.json(); })
      .then(function (r) { if (r.status) apply(r.status); return r; })
      .catch(function (e) { setConn(false, "error: " + e.message); });
  }

  els.toggle.addEventListener("click", function () {
    var running = lastState && lastState.running;
    post({ action: running ? "stop" : "start" });
  });

  $("btn-clear").addEventListener("click", function () { post({ action: "clear_alarm" }); });

  els.tau.addEventListener("input", function () {
    els.tauVal.textContent = Number(els.tau.value).toFixed(2);
  });
  els.tau.addEventListener("change", function () {
    post({ action: "set_tau", value: Number(els.tau.value) });
  });

  els.conf.addEventListener("input", function () {
    els.confVal.textContent = Number(els.conf.value).toFixed(2);
  });
  els.conf.addEventListener("change", function () {
    post({ action: "set_conf", value: Number(els.conf.value) });
  });

  els.policySel.addEventListener("change", function () {
    els.policy.textContent = els.policySel.value;
    post({ action: "set_policy", value: els.policySel.value });
  });

  // ---- input source: camera / video file / YouTube / test pattern --------- //
  function loadMediaIndex() {
    return fetch("/api/media", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (m) {
        els.srcCamera.innerHTML = "";
        (m.cameras || []).forEach(function (c) {
          var o = document.createElement("option");
          o.value = c.index;
          o.textContent = c.name + " (" + c.resolution.join("x") + ")";
          els.srcCamera.appendChild(o);
        });
        els.cameraNote.textContent = (m.cameras || []).length
          ? (m.cameras || []).length + " device(s) detected"
          : "no camera device found — plug one in, press Rescan, or use the test pattern";

        els.srcFile.innerHTML = "";
        (m.files || []).forEach(function (f) {
          var o = document.createElement("option");
          o.value = f.path;
          o.textContent = f.name + "  (" + f.size_mb + " MB)";
          els.srcFile.appendChild(o);
        });
        els.fileNote.textContent = (m.files || []).length
          ? (m.files || []).length + " file(s) in " + m.media_dir
          : "no video files yet — drop one below or type an absolute path";

        els.youtubeNote.textContent = m.youtube_available
          ? "resolved with yt-dlp; the clip is streamed directly, and downloaded only if the stream cannot be decoded"
          : "yt-dlp is not installed — run: pip install yt-dlp";
        return m;
      })
      .catch(function () {
        els.cameraNote.textContent = "could not query the media index";
      });
  }

  function switchSource(kind, value) {
    setConn(true, "switching…");
    return post({ action: "set_source", kind: kind, value: value })
      .then(function (r) {
        if (r && r.error) { setConn(true, "error"); return r; }
        setConn(true, "live");
        return loadMediaIndex();
      });
  }

  Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (tab) {
    tab.addEventListener("click", function () {
      Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (t) {
        t.classList.toggle("on", t === tab);
      });
      Array.prototype.forEach.call(document.querySelectorAll(".tab-body"), function (b) {
        b.classList.toggle("on", b.getAttribute("data-body") === tab.getAttribute("data-tab"));
      });
    });
  });

  Array.prototype.forEach.call(document.querySelectorAll("[data-go]"), function (btn) {
    btn.addEventListener("click", function () {
      var kind = btn.getAttribute("data-go");
      var value = "";
      if (kind === "camera") { value = els.srcCamera.value || "0"; }
      else if (kind === "file") {
        value = (els.srcFilePath.value || "").trim() || els.srcFile.value || "";
        if (!value) { els.fileNote.textContent = "pick a file or type a path first"; return; }
      }
      else if (kind === "youtube") {
        value = (els.srcYoutube.value || "").trim();
        if (!value) { els.youtubeNote.textContent = "paste a YouTube URL first"; return; }
      }
      switchSource(kind, value);
    });
  });

  // ---- uploading a local video -------------------------------------------- //
  function uploadFile(file) {
    if (!file) { return; }
    els.uploadNote.textContent = "uploading " + file.name + "…";
    var fd = new FormData();
    fd.append("file", file, file.name);
    fetch("/api/upload", { method: "POST", body: fd })
      .then(function (r) { return r.json(); })
      .then(function (r) {
        if (r.error) {
          els.uploadNote.textContent = "upload failed: " + r.error;
          return;
        }
        els.uploadNote.textContent = "added " + r.name;
        return loadMediaIndex().then(function () {
          els.srcFile.value = r.path;
          return switchSource("file", r.path);
        });
      })
      .catch(function (e) { els.uploadNote.textContent = "upload failed: " + e.message; });
  }

  els.btnUpload.addEventListener("click", function () { els.fileInput.click(); });
  els.fileInput.addEventListener("change", function () {
    uploadFile(els.fileInput.files && els.fileInput.files[0]);
    els.fileInput.value = "";
  });

  $("btn-rescan").addEventListener("click", function () {
    els.cameraNote.textContent = "scanning…";
    post({ action: "rescan_cameras" });
    var tries = 0;
    var iv = setInterval(function () {
      tries += 1;
      loadMediaIndex();
      if (tries > 12) { clearInterval(iv); }
    }, 1500);
  });

  var viewport = document.querySelector(".viewport");
  ["dragenter", "dragover"].forEach(function (ev) {
    viewport.addEventListener(ev, function (e) {
      e.preventDefault();
      els.dropHint.style.display = "flex";
    });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    viewport.addEventListener(ev, function (e) {
      e.preventDefault();
      els.dropHint.style.display = "none";
    });
  });
  viewport.addEventListener("drop", function (e) {
    var files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) { uploadFile(files[0]); }
  });

  // The MJPEG stream is one long response; if the server restarts, reconnect.
  els.video.addEventListener("error", function () {
    setTimeout(function () { els.video.src = "/video.mjpg?t=" + Date.now(); }, 1500);
  });

  // Attach the stream as soon as the DOM is ready — a never-ending multipart
  // response in the markup would block the load event indefinitely, and waiting
  // for that event would let a slow font CDN stall the live view.
  function attachStream() {
    if (els.video.getAttribute("src")) { return; }   // idempotent
    els.video.src = "/video.mjpg?t=" + Date.now();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attachStream);
  } else {
    attachStream();
  }
  window.addEventListener("load", attachStream);

  poll();
  setInterval(poll, 250);
  loadMediaIndex();
})();
