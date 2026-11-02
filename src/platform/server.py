"""HTTP server for the live platform.

Standard library only (http.server) — no Flask/FastAPI dependency, so the platform
runs with exactly what requirements.txt already installs.

Routes:
    GET  /              the UI (static/index.html)
    GET  /static/<file> UI assets
    GET  /api/status    JSON telemetry (polled by the UI at ~4 Hz)
    GET  /api/config    loaded models, device, thresholds
    POST /api/control   {"action": "start"|"stop"|"set_tau"|"set_conf"|"set_policy"|
                               "clear_alarm", ...}
    GET  /video.mjpg    live annotated stream (multipart/x-mixed-replace)
    GET  /video2.mjpg   live stream of the optional second camera
"""
import json
import mimetypes
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB ceiling for an uploaded clip


def parse_multipart(body: bytes, content_type: str) -> dict:
    """Minimal multipart/form-data parser (stdlib only — cgi is deprecated).

    Returns {field_name: {"filename": str|None, "content_type": str, "data": bytes}}.
    """
    m = re.search(r'boundary="?([^";]+)"?', content_type or "")
    if not m:
        raise ValueError("no multipart boundary in Content-Type")
    boundary = b"--" + m.group(1).encode("utf-8")
    out = {}
    for chunk in body.split(boundary):
        if chunk in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        chunk = chunk.lstrip(b"\r\n")
        if chunk.startswith(b"--"):
            continue
        head, sep, data = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        headers = {}
        for line in head.decode("utf-8", "replace").split("\r\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        disposition = headers.get("content-disposition", "")
        name_m = re.search(r'name="([^"]*)"', disposition)
        file_m = re.search(r'filename="([^"]*)"', disposition)
        if not name_m:
            continue
        out[name_m.group(1)] = {
            "filename": file_m.group(1) if file_m else None,
            "content_type": headers.get("content-type", ""),
            "data": data,
        }
    return out


class MediaIndex:
    """Cached view of what the platform can play.

    Probing camera devices is slow (a missing device blocks OpenCV for tens of
    seconds), so the probe runs in the background and the HTTP layer serves the
    last known result instantly.
    """

    def __init__(self, media_dir, extra_dirs=(), max_cameras=4, refresh_s=60,
                 in_use=None, initial_cameras=None):
        self.media_dir = Path(media_dir)
        self.extra_dirs = [Path(d) for d in extra_dirs]
        self.max_cameras = max_cameras
        self.refresh_s = refresh_s
        self.in_use = in_use          # callable -> indices the platform holds open
        # seeded with a probe taken BEFORE the platform opened its own device
        self.cameras = list(initial_cameras or [])
        self.scanning = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="media-index",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(self.refresh_s)

    def refresh(self):
        from src.platform.camera import probe_cameras

        skip = set()
        if self.in_use is not None:
            try:
                skip = set(self.in_use())
            except Exception:  # noqa: BLE001
                skip = set()
        with self._lock:
            self.scanning = True
        try:
            cams = probe_cameras(max_index=self.max_cameras, per_index_timeout=1.5,
                                 skip=skip)
        except Exception:  # noqa: BLE001 — a failed probe keeps the old list
            cams = None
        finally:
            with self._lock:
                self.scanning = False
        if cams is not None:
            with self._lock:
                # keep the entries we were not allowed to probe
                self.cameras = cams + [c for c in self.cameras if c["index"] in skip]
        return self.cameras

    def files(self):
        from src.platform.camera import list_media

        try:
            return list_media(self.media_dir, self.extra_dirs)
        except Exception:  # noqa: BLE001
            return []

    def snapshot(self):
        with self._lock:
            cameras = list(self.cameras)
            scanning = self.scanning
        try:
            import yt_dlp  # noqa: F401
            youtube = True
        except ImportError:
            youtube = False
        return {"cameras": cameras, "files": self.files(),
                "media_dir": str(self.media_dir.resolve()),
                "youtube_available": youtube, "scanning": scanning}


class PlatformHandler(BaseHTTPRequestHandler):
    server_version = "WildfirePlatform/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------- #
    @property
    def pipeline(self):
        return self.server.pipeline

    def log_message(self, fmt, *args):  # quieter default logging
        if self.server.verbose:
            super().log_message(fmt, *args)

    def _send_bytes(self, data, ctype="application/octet-stream", status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, default=_json_default).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", status)

    def _serve_static(self, rel):
        path = (STATIC_DIR / rel).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), ctype)

    # -- routes ------------------------------------------------------------- #
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        elif path == "/api/status":
            self._send_json(self.pipeline.status())
        elif path == "/api/config":
            self._send_json(self.pipeline.status())
        elif path == "/api/media":
            self._send_json(self._media_index())
        elif path in ("/video.mjpg", "/video2.mjpg"):
            self._stream(which="b" if path == "/video2.mjpg" else "a")
        elif path == "/healthz":
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, 404)

    def _media_index(self):
        """What the UI can offer as an input source: camera devices, video files
        already in the media folder, and whether the YouTube resolver is present.

        Served from the cache — a live probe would stall the page for a minute on
        machines without a camera."""
        index = getattr(self.server, "media_index", None)
        if index is None:
            from src.platform.camera import list_media, probe_cameras
            cfg = getattr(self.server, "media_config", {}) or {}
            media_dir = Path(cfg.get("media_dir", "media"))
            return {"cameras": probe_cameras(max_index=2, per_index_timeout=1.0),
                    "files": list_media(media_dir, cfg.get("extra_dirs", [])),
                    "media_dir": str(media_dir.resolve()),
                    "youtube_available": _youtube_available(),
                    "scanning": False}
        snap = index.snapshot()
        current = (self.pipeline.manager.describe()
                   if hasattr(self.pipeline, "manager") else {})
        snap["current"] = current
        return snap

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/upload":
            self._upload()
            return
        if path != "/api/control":
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            cmd = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"bad request: {e}"}, 400)
            return

        action = str(cmd.get("action", "")).lower()
        p = self.pipeline
        try:
            if action == "start":
                p.start()
            elif action == "stop":
                p.stop()
            elif action == "set_tau":
                p.set_tau(float(cmd["value"]))
            elif action == "set_conf":
                p.set_conf(float(cmd["value"]))
            elif action == "set_policy":
                p.set_policy(str(cmd["value"]))
            elif action == "clear_alarm":
                p.clear_alarm()
            elif action == "set_source":
                kind = str(cmd.get("kind", "")).lower()
                value = cmd.get("value")
                if kind == "youtube" and value:
                    value = str(value).strip()
                p.set_source(kind, value)
            elif action == "rescan_cameras":
                idx = getattr(self.server, "media_index", None)
                if idx is not None:
                    # in the background: a probe of missing devices is slow
                    threading.Thread(target=idx.refresh, daemon=True).start()
            else:
                self._send_json({"error": f"unknown action: {action}"}, 400)
                return
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)
            return
        self._send_json({"ok": True, "status": p.status()})

    # -- upload ------------------------------------------------------------- #
    def _upload(self):
        """Accept a video file (drag & drop in the UI) and store it in the media
        folder, from where it can be selected as a source."""
        from src.platform.camera import VIDEO_EXTS

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json({"error": "empty upload"}, 400)
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json({"error": "file too large (max 2 GB)"}, 413)
            return
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._send_json({"error": "expected multipart/form-data"}, 400)
            return
        body = self.rfile.read(length)
        try:
            fields = parse_multipart(body, ctype)
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"could not parse upload: {e}"}, 400)
            return
        part = fields.get("file")
        if part is None or not part["data"]:
            self._send_json({"error": "no file field in the upload"}, 400)
            return

        filename = Path(part["filename"] or "clip.mp4").name
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTS:
            self._send_json({"error": f"unsupported file type '{suffix}' — "
                                       f"expected one of {', '.join(VIDEO_EXTS)}"}, 400)
            return

        media_dir = Path(getattr(self.server, "media_config", {}).get("media_dir", "media"))
        media_dir.mkdir(parents=True, exist_ok=True)
        target = media_dir / filename
        i = 1
        while target.exists():
            target = media_dir / f"{Path(filename).stem}_{i}{suffix}"
            i += 1
        target.write_bytes(part["data"])
        self._send_json({"ok": True, "name": target.name, "path": str(target.resolve()),
                         "size_mb": round(target.stat().st_size / 1048576, 2)})

    # -- MJPEG -------------------------------------------------------------- #
    def _stream(self, which="a"):
        boundary = "frame"
        self.send_response(200)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        last_seq = -1
        idle = 0
        try:
            while True:
                jpeg, seq = self.pipeline.frame_jpeg(which)
                if jpeg and seq != last_seq:
                    last_seq = seq
                    idle = 0
                    # one write per frame — splitting it makes the client reassemble
                    # the multipart body from several TCP segments
                    block = (f"--{boundary}\r\n"
                             f"Content-Type: image/jpeg\r\n"
                             f"Content-Length: {len(jpeg)}\r\n\r\n").encode()
                    self.wfile.write(block + jpeg + b"\r\n")
                    self.wfile.flush()
                else:
                    idle += 1
                    if idle > 100:          # nothing new for ~10 s -> give up
                        break
                    threading.Event().wait(0.1)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except OSError:
            pass


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(o, set):
        return list(o)
    return str(o)


def _youtube_available():
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


class PlatformServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, pipeline, verbose=False, media_config=None):
        super().__init__(addr, PlatformHandler)
        self.pipeline = pipeline
        self.verbose = verbose
        self.media_config = media_config or {}
        self.media_index = None
        cfg = self.media_config
        if cfg:
            self.media_index = MediaIndex(
                media_dir=cfg.get("media_dir", "media"),
                extra_dirs=cfg.get("extra_dirs", []),
                max_cameras=int(cfg.get("max_cameras", 4)),
                refresh_s=float(cfg.get("refresh_s", 60)),
                in_use=(getattr(pipeline, "active_camera_indices", None)),
                initial_cameras=cfg.get("cameras"),
            ).start()

    def handle_error(self, request, client_address):
        """A browser closing the MJPEG stream is normal, not an error worth a
        traceback on the console."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def serve(pipeline, host="127.0.0.1", port=8080, verbose=False, media_config=None):
    """Blocking: runs the server until interrupted."""
    httpd = PlatformServer((host, port), pipeline, verbose=verbose,
                           media_config=media_config)
    print(f"[platform] serving on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[platform] shutting down")
    finally:
        httpd.shutdown()
        httpd.server_close()
        if httpd.media_index is not None:
            httpd.media_index.stop()
        pipeline.stop()
    return httpd
