"""§3.4 — Fetch the three benchmarks.

Usage:
    python scripts/get_datasets.py --all
    python scripts/get_datasets.py --dataset dfire|figlib|flame
    python scripts/get_datasets.py --dataset figlib --subset 5   # N sequences, smoke test

For each dataset:
  1. Check data/<name>/.done marker; skip if present.
  2. dfire:  the gaiasd git repo no longer ships the data (README/utils only). Download
             the Kaggle mirror the official README links to via kagglehub
             (sayedgamal99/smoke-fire-detection-yolo) -> organize into
             data/dfire/{images,labels}/{train,val,test}, remapping class ids to the
             canonical {0: smoke, 1: fire} taxonomy if the mirror differs.
  3. figlib: download the official split manifest from the gitlab snippet (currently a
             plain-text list of sequence names), then resolve each sequence against
             HPWREN's CDN index (https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/ — the
             old c1.hpwren.ucsd.edu/archive host is dead) and fetch only the listed
             images. Use --subset N to smoke-test the pipeline before pulling everything.
  4. flame:  fetch the FLAME 3 Computer Vision subset (Sycan Marsh single burn, 738
             RGB + 738 thermal pairs) from Kaggle via kagglehub — no IEEE DataPort
             account needed — and re-root it into data/flame/{fire,none} class dirs.
             The full 6-burn FLAME 3 / FLAME1 / FLAME2 sets still live behind IEEE
             DataPort (account + license); their manual steps are printed if the
             Kaggle mirror is unavailable.
  5. Write data/<name>/.done with a timestamp + source URL + file count for provenance.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

DFIRE_REPO = "https://github.com/gaiasd/DFireDataset"
FIGLIB_MANIFEST_URLS = [
    "https://gitlab.nrp-nautilus.io/-/snippets/63/raw/main/figlib_cv_splits.json",
    "https://gitlab.nrp-nautilus.io/-/snippets/63/raw",
]
# The old c1.hpwren.ucsd.edu archive host is dead; HPWREN's FIgLib page (updated
# 2024-12) now serves the sequences from the CDN below, one <seq>/index.html each.
FIGLIB_CDN_BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/"
FLAME_PAGE = ("https://ieee-dataport.org/open-access/"
              "flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs")

EXTS = (".jpg", ".jpeg", ".png")


def write_done(name: str, source: str, count: int):
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    (out / ".done").write_text(
        f"ts={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"source={source}\nfiles={count}\n", encoding="utf-8")
    print(f"[{name}] .done written ({count} files)")


def count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in EXTS)


# --------------------------------------------------------------------------- #
DFIRE_KAGGLE_SLUG = "sayedgamal99/smoke-fire-detection-yolo"  # official mirror linked
                                                              # in the D-Fire README
CANONICAL = ["smoke", "fire"]  # configs/datasets.yaml taxonomy (0=smoke, 1=fire)


def _dfire_class_map(src: Path):
    """Return a dict {source_id: canonical_id} for the mirror's labels.

    Reads the mirror's own data.yaml if it ships one; otherwise keeps ids as-is
    and says so loudly (verify afterwards via harmonize_labels.py's histogram
    against the published counts: fire 14,692 > smoke 11,865).
    """
    names = None
    for y in sorted(src.rglob("*.yaml")):
        try:
            d = yaml.safe_load(y.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — probe, report and continue
            continue
        if isinstance(d, dict) and "names" in d:
            names = d["names"]
            print(f"[dfire] mirror {y.relative_to(src)} declares names={names}")
            break
    if names is None:
        print("[dfire] no data.yaml found in mirror — keeping source class ids "
              "UNVERIFIED; run scripts/harmonize_labels.py --dataset dfire and check "
              "the histogram against fire 14,692 / smoke 11,865.")
        return {0: 0, 1: 1}
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if names[:2] == CANONICAL:
        return {0: 0, 1: 1}
    if names[:2] == list(reversed(CANONICAL)):
        print("[dfire] mirror uses 0=fire,1=smoke — remapping to canonical "
              "0=smoke,1=fire while copying labels.")
        return {0: 1, 1: 0}
    print(f"[dfire] mirror names {names} don't match the canonical pair — keeping "
          "ids as-is; verify with harmonize_labels.py before training.")
    return {0: 0, 1: 1}


def _dfire_from_kaggle(out: Path) -> tuple[int, str] | None:
    """Fetch D-Fire via the Kaggle mirror the official README links to.

    The upstream git repo no longer ships the data (only README/utils — the dataset
    moved to OneDrive + mirrors), so organize whatever layout the mirror uses into
    data/dfire/{images,labels}/{train,val,test} (ultralytics convention).
    Returns (moved_images, provenance) or None on failure.
    """
    try:
        import kagglehub
    except ImportError:
        print("[dfire] kagglehub is not installed (it is in requirements.txt)")
        return None
    try:
        src = Path(kagglehub.dataset_download(DFIRE_KAGGLE_SLUG))
    except Exception as e:  # noqa: BLE001 — network probing, report and continue
        print(f"[dfire] kagglehub {DFIRE_KAGGLE_SLUG}: {e}")
        return None
    print(f"[dfire] mirror downloaded to {src}")

    cmap = _dfire_class_map(src)
    hist = {name: 0 for name in CANONICAL}
    moved = 0
    for img in sorted(src.rglob("*")):
        if img.suffix.lower() not in EXTS or not img.is_file():
            continue
        parts = [p.lower() for p in img.parts]
        split = next(({"valid": "val", "validation": "val"}.get(p, p)
                      for p in reversed(parts) if p in ("train", "val", "valid",
                                                        "validation", "test")), None)
        if split is None:  # flat mirror: build_splits.py re-splits anyway
            split = "train"
        img_dst = out / "images" / split / img.name
        if img_dst.exists():
            continue
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        lbl_src = None
        for cand in (img.with_suffix(".txt"),                          # alongside
                     img.parent.parent / "labels" / (img.stem + ".txt"),   # <x>/images/.. -> <x>/labels/
                     img.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")):
            if cand.exists():
                lbl_src = cand
                break
        shutil.copy2(img, img_dst)
        if lbl_src is not None:
            lines_out = []
            for line in lbl_src.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                tok = line.split()
                cid = int(tok[0])
                if cid in cmap:
                    tok[0] = str(cmap[cid])
                    hist[CANONICAL[cmap[cid]]] += 1
                lines_out.append(" ".join(tok))
            (out / "labels" / split / (img.stem + ".txt")).write_text(
                "\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
        moved += 1
    if moved:
        print(f"[dfire] canonical instance histogram: {hist} "
              f"(published full set: fire=14692, smoke=11865)")
    return (moved, f"kagglehub:{DFIRE_KAGGLE_SLUG}")


def get_dfire() -> bool:
    out = DATA / "dfire"
    out.mkdir(parents=True, exist_ok=True)

    fetched = _dfire_from_kaggle(out)
    if fetched is None:
        print("[dfire] Kaggle mirror unavailable. Manual options:\n"
              "  1. Kaggle: https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo\n"
              "     (download the zip, unzip into data/dfire/ keeping its images/ +\n"
              "     labels/ layout, delete data/dfire/.done, re-run this script)\n"
              f"  2. OneDrive (official links): open {DFIRE_REPO} README.\n"
              "  NOTE: the gaiasd git repo no longer ships the dataset — cloning it "
              "yields no images.")
        return False
    moved, source = fetched

    if moved == 0:
        print("[dfire] nothing copied from the mirror — inspect the download")
        return False
    write_done("dfire", source, moved)
    return True


# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout: int = 60) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001 — network probing, report and continue
        print(f"  ! {url} -> {e}")
        return None


def _figlib_sequences_from_text(blob: bytes) -> list[str]:
    """The official splits snippet currently serves one sequence name per line
    (e.g. 20170708_Whittier_syp-n-mobo-m) rather than JSON."""
    names = []
    for line in blob.decode("utf-8", "replace").splitlines():
        line = line.strip().split("/")[-1]
        if re.fullmatch(r"\d{8}_\S+", line):
            names.append(line)
    return names


def _figlib_resolve(seq: str, avail: set[str]) -> str | None:
    """Exact index match, else same date+fire prefix (camera-model suffixes drift)."""
    if not avail or seq in avail:
        return seq
    prefix = "_".join(seq.split("_")[:2])
    cands = [a for a in avail if a.startswith(prefix)]
    if not cands:
        return None
    tail = seq.rsplit("-", 1)[-1]
    same = [c for c in cands if c.rsplit("-", 1)[-1] == tail]
    pick = (same or cands)[0]
    print(f"[figlib] '{seq}' not on CDN verbatim — using '{pick}'")
    return pick


def _figlib_sequence_urls(seq: str) -> list[str]:
    page = _http_get(FIGLIB_CDN_BASE + seq + "/index.html")
    if page is None:
        return []
    hrefs = re.findall(r'href="?([^">\s]+?\.(?:jpg|jpeg|png))"?',
                       page.decode("utf-8", "replace"), re.IGNORECASE)
    return [FIGLIB_CDN_BASE + seq + "/" + h for h in hrefs]


def get_figlib(subset: int | None = None) -> bool:
    out = DATA / "figlib"
    out.mkdir(parents=True, exist_ok=True)

    seqs = []
    manifest = None
    for url in FIGLIB_MANIFEST_URLS:
        print(f"[figlib] trying split manifest {url} ...")
        blob = _http_get(url)
        if not blob:
            continue
        try:
            manifest = json.loads(blob)
            print(f"[figlib] fetched JSON manifest from {url}")
            break
        except json.JSONDecodeError:
            (out / "splits_raw.txt").write_bytes(blob)
            seqs = _figlib_sequences_from_text(blob)
            if seqs:
                print(f"[figlib] text manifest with {len(seqs)} sequence names "
                      f"saved to data/figlib/splits_raw.txt")
                break
            print(f"[figlib] {url}: neither JSON nor a sequence list — inspect "
                  "data/figlib/splits_raw.txt manually")

    if manifest is not None:
        # JSON path: the manifest itself carries image URLs.
        def iter_image_urls(node):
            if isinstance(node, str) and node.lower().split("?")[0].endswith(EXTS):
                yield node
            elif isinstance(node, dict):
                for v in node.values():
                    yield from iter_image_urls(v)
            elif isinstance(node, list):
                for v in node:
                    yield from iter_image_urls(v)
        urls = list(dict.fromkeys(iter_image_urls(manifest)))
    elif seqs:
        # Text path: resolve each sequence against the CDN index, then list its images.
        index = _http_get(FIGLIB_CDN_BASE + "index.html")
        avail = set(re.findall(r'href="?(\S+?)/index\.html"?',
                               index.decode("utf-8", "replace"))) if index else set()
        urls = []
        for s in seqs:
            got = _figlib_resolve(s, avail)
            if got is None:
                print(f"[figlib] '{s}' has no CDN counterpart — skipped")
                continue
            seq_urls = _figlib_sequence_urls(got)
            if not seq_urls:
                print(f"[figlib] '{got}': no images listed — skipped")
                continue
            urls.extend(seq_urls)
            if subset and len(urls) >= subset:
                break
    else:
        urls = []

    if not urls:
        print("[figlib] no image URLs resolved — fetch the official splits manually: "
              "https://gitlab.nrp-nautilus.io/-/snippets/63 and the archive index: "
              + FIGLIB_CDN_BASE)
        return False
    if subset:
        urls = urls[:subset]

    n_ok = 0
    for u in urls:
        rel = u.split("HPWREN-FIgLib-Data/")[-1].strip("/")  # <seq>/<file>.jpg
        dst = out / "images" / rel
        if dst.exists():
            n_ok += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob = _http_get(u, timeout=120)
        if blob:
            dst.write_bytes(blob)
            n_ok += 1
    write_done("figlib", FIGLIB_CDN_BASE, n_ok)
    print("[figlib] NOTE: these are fire sequences; negative windows come from the "
          "official CV splits — keep them sequence-level to prevent leakage (§3.5).")
    return n_ok > 0


# --------------------------------------------------------------------------- #
FLAME3_KAGGLE_SLUG = "brycehopkins/flame-3-computer-vision-subset-sycan-marsh"


def _flame_label_from_parts(parts: tuple) -> str | None:
    """Class-dir detection mirroring build_splits.collect_class_dirs /
    harmonize_labels.report_classification_tree (case-insensitive)."""
    pl = [p.lower() for p in parts]
    if any(p == "fire" for p in pl):
        return "fire"
    if any("non" in p and "fire" in p for p in pl):
        return "none"
    if any(p in ("no fire", "no_fire", "nofire", "nonfire", "none", "negative")
           for p in pl):
        return "none"
    return None


def _flame_from_kaggle(out: Path) -> tuple[int, int, str] | None:
    """Fetch the FLAME 3 CV subset (Sycan Marsh single burn, 738 RGB + 738 thermal
    pairs) from Kaggle — no IEEE DataPort account required.

    Re-roots the mirror's class-dir tree at data/flame/{fire,none}/ keeping the
    RGB/Thermal substructure (downstream expects the classification tree; the
    event grouping for leakage-safe splits comes from the filename stem).
    Returns (n_rgb, n_thermal, provenance) or None on failure.
    """
    try:
        import kagglehub
    except ImportError:
        print("[flame] kagglehub is not installed (it is in requirements.txt)")
        return None
    try:
        src = Path(kagglehub.dataset_download(FLAME3_KAGGLE_SLUG))
    except Exception as e:  # noqa: BLE001 — network probing, report and continue
        print(f"[flame] kagglehub {FLAME3_KAGGLE_SLUG}: {e}")
        return None
    print(f"[flame] mirror downloaded to {src}")

    n_rgb = n_thermal = 0
    skipped = 0
    for img in sorted(src.rglob("*")):
        if img.suffix.lower() not in EXTS or not img.is_file():
            continue
        label = _flame_label_from_parts(img.relative_to(src).parts)
        if label is None:
            skipped += 1
            continue
        rel = img.relative_to(src)
        dst = out / label / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst)
        if "thermal" in [p.lower() for p in rel.parts] or img.suffix.lower() in (".tif", ".tiff"):
            n_thermal += 1
        else:
            n_rgb += 1
    print(f"[flame] copied {n_rgb} visual + {n_thermal} thermal images into "
          f"data/flame/{{fire,none}}" + (f" ({skipped} images outside class dirs skipped)"
                                         if skipped else ""))
    return (n_rgb, n_thermal, f"kagglehub:{FLAME3_KAGGLE_SLUG}")


def get_flame() -> bool:
    out = DATA / "flame"
    out.mkdir(parents=True, exist_ok=True)

    fetched = _flame_from_kaggle(out)
    if fetched is None:
        print("[flame] Kaggle mirror unavailable. The full 6-burn FLAME 3 set and the "
              "older FLAME1/FLAME2 still require IEEE DataPort (account + license):\n"
              f"  1. Create a free account at {FLAME_PAGE}\n"
              "  2. Accept the license and download the FLAME zips (RGB + LWIR)\n"
              f"  3. Unzip into {out}\n"
              "  4. Re-run this script to write the .done marker.\n"
              "Optional: export IEEE_DATAPORT_COOKIE='...' for an authenticated page probe.")
        return False
    n_rgb, n_thermal, source = fetched
    if n_rgb + n_thermal == 0:
        print("[flame] nothing copied — mirror layout has no recognizable class dirs; "
              "inspect the download and extend _flame_label_from_parts")
        return False
    write_done("flame", source, n_rgb + n_thermal)
    return True


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="§3 — fetch FLAME / D-Fire / FIgLib")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", choices=["flame", "dfire", "figlib"])
    ap.add_argument("--subset", type=int, default=None,
                    help="figlib: fetch only N sequences (smoke test)")
    args = ap.parse_args()

    wanted = ["dfire", "figlib", "flame"] if args.all else ([args.dataset] if args.dataset else [])
    if not wanted:
        ap.error("pass --all or --dataset")

    for name in wanted:
        marker = DATA / name / ".done"
        if marker.exists():
            print(f"[{name}] .done present — skipping (delete it to force re-fetch)")
            continue
        print(f"=== {name} ===")
        ok = {"dfire": get_dfire, "figlib": get_figlib, "flame": get_flame}[name](
            **({"subset": args.subset} if name == "figlib" else {}))
        if not ok:
            print(f"[{name}] INCOMPLETE — see messages above.")


if __name__ == "__main__":
    sys.exit(main())
