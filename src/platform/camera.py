"""Frame sources for the live platform.

Sources, all exposing the same `read()` contract and all swappable at runtime
through `SourceManager`:

  * "camera"   — the PC webcam through OpenCV (DirectShow first on Windows, because
                 the MSMF backend often opens the wrong device or blocks for seconds)
  * "file"     — a local video file (mp4 / avi / mov / mkv / webm) replayed in a loop
  * "youtube"  — a YouTube URL, resolved with yt-dlp: the direct media URL is
                 streamed straight into OpenCV, and if that fails the video is
                 downloaded into the media folder and played from disk
  * "synthetic"— an animated test pattern (no camera needed: smoke-test the plumbing,
                 and gives the browser UI something to render)

Capture runs in its own thread so a slow disk, a stalled driver or a slow YouTube
resolve never blocks the inference loop; the most recent frame is kept in a
one-slot buffer.
"""
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv")

_cv2_module = None

# Serialises every device open. Two handles on one DirectShow device can stall
# the driver's read path, so a device is never opened twice at the same time.
_DEVICE_OPEN_LOCK = threading.Lock()


def _cv2():
    """Import cv2 once, with its logger silenced.

    Two separate noises are suppressed:
      * OpenCV's backend notice ("should be configured with libavdevice…") is
        printed by the C++ library the first time cv2 is imported, before any
        Python-side log level applies — it is a red herring, since the platform
        captures through DirectShow, not through FFmpeg
      * the device-probe spam about missing indices, which is expected

    Real failures are reported through `FrameSource.error` and shown in the UI.
    """
    global _cv2_module
    if _cv2_module is not None:
        return _cv2_module

    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        try:
            import cv2
        finally:
            os.dup2(saved, 2)
    finally:
        os.close(devnull)
        os.close(saved)
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:  # noqa: BLE001 — older builds may not expose the logger
        pass
    _cv2_module = cv2
    return cv2


class FrameSource:
    """Base class: background capture thread + one-slot latest-frame buffer."""

    def __init__(self, width=640, height=480, fps=30, name="source"):
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self.kind = "base"
        self._lock = threading.Lock()
        self._frame = None
        self._ts = 0.0
        self._seq = 0
        self._stop = threading.Event()
        self._thread = None
        self.error = None
        self.status = ""          # human-readable progress, shown in the UI
        self.device_info = ""

    # -- lifecycle ---------------------------------------------------------- #
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"src-{self.kind}",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def _run(self):
        raise NotImplementedError

    # -- access ------------------------------------------------------------- #
    def read(self):
        """Returns (frame_or_None, timestamp, seq). Frame is a BGR uint8 array."""
        with self._lock:
            return self._frame, self._ts, self._seq

    def _publish(self, frame, ts=None):
        with self._lock:
            self._frame = frame
            self._ts = time.time() if ts is None else ts
            self._seq += 1

    def describe(self):
        return {"kind": self.kind, "name": self.name, "info": self.device_info,
                "status": self.status, "error": self.error}


class CameraSource(FrameSource):
    """PC webcam capture."""

    def __init__(self, index=0, width=640, height=480, fps=30, backend="auto"):
        super().__init__(width=width, height=height, fps=fps, name=f"camera{index}")
        self.kind = "camera"
        self.index = index
        self.backend = backend

    def _open(self):
        cv2 = _cv2()

        backends = []
        if self.backend == "auto":
            # DirectShow is the most reliable on Windows; MSMF is the portable fallback.
            if hasattr(cv2, "CAP_DSHOW"):
                backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
            else:
                backends = [cv2.CAP_ANY]
        elif self.backend == "dshow":
            backends = [cv2.CAP_DSHOW]
        elif self.backend == "msmf":
            backends = [cv2.CAP_MSMF]
        else:
            backends = [cv2.CAP_ANY]

        last_err = None
        for be in backends:
            with _DEVICE_OPEN_LOCK:
                cap = cv2.VideoCapture(self.index, be)
                if not cap.isOpened():
                    last_err = f"index {self.index} did not open with backend {be}"
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                # give the driver a moment before the first read
                frame = None
                for _ in range(10):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        break
            if frame is not None:
                self.device_info = (f"index {self.index}, backend {be}, "
                                    f"{frame.shape[1]}x{frame.shape[0]}")
                return cap
            cap.release()
            last_err = f"index {self.index} opened but returned no frames"
        raise RuntimeError(last_err or "camera could not be opened")

    def _run(self):
        try:
            cap = self._open()
        except Exception as e:  # noqa: BLE001 — surfaced to the UI, not fatal
            self.error = str(e)
            self.status = "camera unavailable"
            return
        self.status = "live"
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if ok and frame is not None:
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = _resize(frame, self.width, self.height)
                    self._publish(frame)
                else:
                    time.sleep(0.005)
        finally:
            cap.release()


class FileSource(FrameSource):
    """Replay a local video file in a loop.

    `path` may be a single file or a directory (the first video inside it is
    used), which makes pointing the platform at a folder of clips easy.
    """

    def __init__(self, path, width=640, height=480, fps=30, loop=True):
        super().__init__(width=width, height=height, fps=fps, name=Path(path).name)
        self.kind = "file"
        self.path = Path(path)
        self.loop = loop

    def _resolve(self):
        p = self.path
        if p.is_dir():
            found = sorted(f for f in p.iterdir()
                           if f.suffix.lower() in VIDEO_EXTS)
            if not found:
                raise FileNotFoundError(f"no video file found in {p}")
            p = found[0]
        if not p.exists():
            raise FileNotFoundError(f"video file not found: {p}")
        return p

    def _run(self):
        cv2 = _cv2()
        try:
            path = self._resolve()
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            self.status = "file unavailable"
            return
        self.device_info = str(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.error = f"could not open video file: {path}"
            self.status = "file unavailable"
            return
        src_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        self.fps = float(src_fps) if 1 <= src_fps <= 240 else self.fps
        self.status = "playing"
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if not self.loop:
                        self.status = "ended"
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = _resize(frame, self.width, self.height)
                self._publish(frame)
                time.sleep(1.0 / max(self.fps, 1))
        finally:
            cap.release()


class YouTubeSource(FrameSource):
    """Play a YouTube video as a live frame source.

    Resolution order:
      1. yt-dlp resolves the URL to a direct video URL, which OpenCV streams over
         HTTP — fast start, no disk copy
      2. if OpenCV cannot open that URL (codec / network / container), the video
         is downloaded into the media folder and played from disk

    Only the VIDEO track is taken: YouTube serves audio and video as separate
    DASH streams, and a detector does not need audio. Asking for `bestvideo`
    therefore gets a plain, seekable mp4 that FFmpeg can read over HTTPS — and
    halves the transfer.

    The direct URL is time-limited, so the source re-resolves it whenever the
    stream ends.

    yt-dlp is optional: without it this source reports a clear error instead of
    failing at import time.
    """

    # video-only: a detector never needs the audio track
    FORMAT = "bestvideo[ext=mp4][height<=1080]/bestvideo[ext=mp4]/bestvideo"

    def __init__(self, url, width=640, height=480, fps=30, media_dir=None):
        super().__init__(width=width, height=height, fps=fps, name="youtube")
        self.kind = "youtube"
        self.url = url
        self.media_dir = Path(media_dir) if media_dir else Path("media")
        self.title = ""
        self.is_live = False

    # -- resolution --------------------------------------------------------- #
    @staticmethod
    def _yt_dlp():
        try:
            import yt_dlp
        except ImportError:
            raise RuntimeError(
                "YouTube source needs yt-dlp: pip install yt-dlp")
        return yt_dlp

    def _resolve_url(self):
        """-> (direct_url, info_dict)"""
        yt_dlp = self._yt_dlp()
        opts = {
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "skip_download": True,
            "format": self.FORMAT,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(self.url, download=False)
        if not info:
            raise RuntimeError(f"yt-dlp could not resolve {self.url}")
        self.title = info.get("title") or ""
        direct = info.get("url")
        if not direct:
            raise RuntimeError(f"no direct media URL for {self.url}")
        return direct, info

    def _download(self, url):
        """Fallback: download into the media folder, return the local path."""
        yt_dlp = self._yt_dlp()
        self.media_dir.mkdir(parents=True, exist_ok=True)
        outtmpl = str(self.media_dir / "%(id)s.%(ext)s")
        opts = {
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "format": self.FORMAT,
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
            "progress_hooks": [self._progress],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        if not Path(path).exists():
            candidates = sorted(self.media_dir.glob(f"{info.get('id', '*')}.*"))
            if not candidates:
                raise RuntimeError("download finished but no file was produced")
            path = str(candidates[0])
        return path

    def _progress(self, d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                self.status = f"downloading {done * 100 // total}%"
        elif d.get("status") == "finished":
            self.status = "downloaded"

    def _open(self):
        cv2 = _cv2()
        self.status = "resolving"
        try:
            direct, info = self._resolve_url()
        except Exception as e:  # noqa: BLE001
            self.error = f"yt-dlp could not resolve this URL: {e}"
            raise
        self.is_live = bool(info.get("is_live"))
        self.device_info = (f"{self.title or self.url} "
                            f"({'live' if self.is_live else 'video'})")

        self.status = "connecting"
        cap = cv2.VideoCapture(direct)
        ok = False
        frame = None
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        if ok and frame is not None:
            self.status = "streaming"
            return cap
        cap.release()

        # fallback: download and play locally
        self.status = "downloading"
        path = self._download(self.url)
        self.device_info = f"{self.title or self.url} (downloaded)"
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open the downloaded video: {path}")
        self.status = "playing"
        return cap

    def _run(self):
        """Play, and re-resolve when the (time-limited) direct URL expires.

        A VOD clip is replayed from the start when it ends, so a short video can
        be used as a repeating test input. A live stream is not looped — when it
        ends, it has really ended.
        """
        attempts = 0
        while not self._stop.is_set():
            try:
                cap = self._open()
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                self.status = "unavailable"
                return
            last_ok = time.time()
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        last_ok = time.time()
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = _resize(frame, self.width, self.height)
                        self._publish(frame)
                    else:
                        if time.time() - last_ok > 3.0:
                            break          # the stream ended (or the URL expired)
                        time.sleep(0.05)
            finally:
                cap.release()
            attempts += 1
            if self._stop.is_set():
                return
            if self.is_live:
                self.status = "ended"
                return
            self.status = "replaying"
            time.sleep(1.0)


class SyntheticSource(FrameSource):
    """Animated test pattern: sky, ground, a rising smoke plume and a flickering fire
    base. It exists so the platform can be smoke-tested (and demoed) on a machine
    with no camera. The frames are SYNTHETIC — they are never a detection claim."""

    def __init__(self, width=640, height=480, fps=30, plume=True, fire=True):
        super().__init__(width=width, height=height, fps=fps, name="synthetic")
        self.kind = "synthetic"
        self.plume = plume
        self.fire = fire
        self.device_info = "synthetic test pattern (SYNTHETIC frames)"

    def _render(self, t):
        import cv2

        w, h = self.width, self.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # sky: vertical gradient
        for y in range(h):
            v = int(150 + 90 * (1 - y / h))
            frame[y, :, :] = (v, int(v * 0.95), int(v * 0.85))
        # ground
        ground_y = int(h * 0.72)
        frame[ground_y:, :] = (58, 92, 74)
        # horizon texture
        rng = np.random.default_rng(7)
        for _ in range(60):
            x = int(rng.integers(0, w))
            y = int(rng.integers(ground_y, h))
            frame[y, x] = (46, 78, 62)

        cx = int(w * 0.42)
        base_y = ground_y

        if self.fire:
            # flickering fire base
            flick = 0.85 + 0.15 * np.sin(t * 11.0) + 0.05 * np.sin(t * 27.0)
            fh = int(28 * flick)
            fw = int(34 * flick)
            cv2.ellipse(frame, (cx, base_y - fh // 2), (fw // 2, fh // 2), 0, 0, 360,
                        (40, 60, 235), -1)
            cv2.ellipse(frame, (cx, base_y - fh // 2), (fw // 4, fh // 3), 0, 0, 360,
                        (30, 210, 255), -1)

        if self.plume:
            # rising, expanding grey plume
            for i in range(16):
                p = i / 15.0
                y = base_y - int(30 + 210 * p)
                spread = int(16 + 58 * p)
                wob = int(10 * np.sin(t * 1.7 + i * 0.55))
                a = 0.45 * (1.0 - p * 0.55)
                overlay = frame.copy()
                cv2.circle(overlay, (cx + wob, y), spread, (128, 128, 128), -1)
                frame = cv2.addWeighted(overlay, a, frame, 1 - a, 0)

        # frame counter strip (visual only)
        cv2.rectangle(frame, (0, 0), (168, 22), (30, 30, 30), -1)
        cv2.putText(frame, f"SYNTHETIC  t={t:6.2f}s", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        return frame

    def _run(self):
        t0 = time.time()
        period = 1.0 / max(self.fps, 1)
        self.status = "live"
        while not self._stop.is_set():
            t = time.time() - t0
            self._publish(self._render(t))
            time.sleep(period)


def _resize(frame, w, h):
    import cv2

    return cv2.resize(frame, (w, h))


def _probe_one(index, timeout):
    """Open one device index with a hard timeout.

    Two things matter here:
      * the DirectShow backend is tried first — it fails instantly on an index
        that does not exist, whereas the default (MSMF) backend enumerates every
        device and can block for a minute per index
      * the open runs in a throw-away thread and takes the shared device lock, so
        a probe can never open a device the platform is already using

    OpenCV has no timeout of its own, hence the thread + join.
    """
    out = {}

    def work():
        cv2 = _cv2()
        backends = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
        for be in backends:
            with _DEVICE_OPEN_LOCK:
                cap = cv2.VideoCapture(index, be)
                try:
                    if not cap.isOpened():
                        continue
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        out["index"] = index
                        out["resolution"] = [int(frame.shape[1]), int(frame.shape[0])]
                        out["name"] = f"camera {index}"
                        return
                finally:
                    cap.release()

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    return out or None


def probe_cameras(max_index=6, per_index_timeout=1.5, skip=()):
    """Try to open device indices 0..max_index-1 and report which ones work.

    Used by `run_platform.py --list-cameras` and by the UI's device picker so the
    user can pick the right index for their PC (Windows often has the built-in
    webcam at 0 and an external USB camera at 1 or 2).

    Each index is probed under a hard timeout — a missing device must not stall
    the caller for a minute. Indices in `skip` are left alone: they are in use by
    the running platform, and a second handle on the same device can stall the
    live capture.
    """
    skip = set(skip or ())
    found = []
    for i in range(max_index):
        if i in skip:
            continue
        try:
            res = _probe_one(i, per_index_timeout)
        except Exception:  # noqa: BLE001 — probing must never crash the caller
            res = None
        if res:
            found.append(res)
    return found


def list_media(media_dir, extra_dirs=()):
    """Video files available to the platform: the media folder plus any extra
    directories, each returned with its absolute path so the UI can pick it."""
    out = []
    seen = set()
    dirs = [Path(media_dir)] + [Path(d) for d in extra_dirs]
    for d in dirs:
        if not d or not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in VIDEO_EXTS:
                continue
            key = str(f.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": f.name, "path": str(f.resolve()),
                        "size_mb": round(f.stat().st_size / 1048576, 2)})
    return out


def make_source(kind, value=None, width=640, height=480, fps=30, backend="auto",
                media_dir=None):
    """Build a frame source from a (kind, value) pair — the UI and the CLI speak
    the same vocabulary."""
    kind = (kind or "").lower()
    if kind == "camera":
        idx = int(value) if value not in (None, "") else 0
        return CameraSource(index=idx, width=width, height=height, fps=fps,
                            backend=backend)
    if kind == "file":
        if not value:
            raise ValueError("a file path is required")
        return FileSource(value, width=width, height=height, fps=fps)
    if kind == "youtube":
        if not value:
            raise ValueError("a YouTube URL is required")
        return YouTubeSource(value, width=width, height=height, fps=fps,
                             media_dir=media_dir)
    if kind == "synthetic":
        return SyntheticSource(width=width, height=height, fps=fps)
    raise ValueError(f"unknown source kind: {kind}")


def build_source(source="camera", index=0, path=None, width=640, height=480, fps=30,
                 backend="auto"):
    """Backwards-compatible helper used by scripts/run_platform.py."""
    if source == "camera":
        return make_source("camera", index, width=width, height=height, fps=fps,
                           backend=backend)
    if source == "file":
        return make_source("file", path, width=width, height=height, fps=fps)
    if source == "synthetic":
        return make_source("synthetic", width=width, height=height, fps=fps)
    raise ValueError(f"unknown source: {source}")


def is_youtube_url(url):
    """Cheap check for a YouTube link (no network, no yt-dlp needed)."""
    if not url:
        return False
    return bool(re.search(r"(youtube\.com|youtu\.be)", str(url), re.IGNORECASE))


class SourceManager:
    """Holds the active frame source and swaps it at runtime.

    The inference loop always reads through the manager, so switching from the
    live camera to an MP4 (or a YouTube link) does not restart the model, the
    fusion loop or the HTTP server.
    """

    def __init__(self, source, secondary=None):
        self._lock = threading.RLock()
        self.source = source
        self.secondary = secondary
        self._seq = 0
        self._last_src_seq = -1

    @property
    def current(self):
        with self._lock:
            return self.source

    def read(self):
        """-> (frame|None, timestamp, global_seq).

        The seq only advances when the source published a NEW frame, so the
        inference loop can skip duplicates instead of re-running the model on the
        same image. It stays monotonic across a source swap.
        """
        with self._lock:
            src = self.source
        if src is None:
            return None, 0.0, self._seq
        frame, ts, src_seq = src.read()
        if frame is not None and src_seq != self._last_src_seq:
            self._last_src_seq = src_seq
            self._seq += 1
        return frame, ts, self._seq

    def swap(self, new_source):
        with self._lock:
            old = self.source
            self.source = new_source
            self._last_src_seq = -1
        if old is not None and old is not new_source:
            old.stop()
        new_source.start()
        return new_source

    def start(self):
        with self._lock:
            src, sec = self.source, self.secondary
        if src is not None:
            src.start()
        if sec is not None:
            sec.start()

    def stop(self):
        with self._lock:
            src, sec = self.source, self.secondary
        if src is not None:
            src.stop()
        if sec is not None:
            sec.stop()

    def describe(self):
        with self._lock:
            src = self.source
        return src.describe() if src is not None else {"kind": "none", "name": "-",
                                                       "info": "", "status": "", "error": None}
