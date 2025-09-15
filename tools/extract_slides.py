#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import timedelta

from PIL import Image
import imagehash


def hhmmss(seconds: float) -> str:
    td = timedelta(seconds=float(seconds))
    # Drop microseconds for cleaner labels
    s = str(td).split(".")[0]
    if td.days > 0:
        # Normalize day offset (shouldn't happen for lecture videos)
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        sec = total % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"
    # Ensure HH:MM:SS
    parts = s.split(":")
    if len(parts) == 2:
        return f"00:{s}"
    return s


def run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def extract_candidates(video_path: str, out_dir: str, scene_threshold: float) -> tuple[list[str], list[float]]:
    candidates_dir = os.path.join(out_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "ffmpeg_showinfo.log")
    # Use showinfo to get pts_time; select scene changes for candidate frames
    vf = f"select='gt(scene,{scene_threshold})',showinfo"
    out_pattern = os.path.join(candidates_dir, "%06d.png")
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", video_path,
        "-vf", vf,
        "-vsync", "vfr",
        out_pattern,
    ]
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed; see log at {log_path}")

    # Parse pts_time from showinfo log
    pts_times: list[float] = []
    pts_re = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")
    with open(log_path, "r") as f:
        for line in f:
            m = pts_re.search(line)
            if m:
                try:
                    pts_times.append(float(m.group(1)))
                except ValueError:
                    pass

    # Gather only the sequence produced by this run: 000001.png..N
    files = []
    for i in range(1, len(pts_times) + 1):
        p = os.path.join(candidates_dir, f"{i:06d}.png")
        if os.path.exists(p):
            files.append(p)
    # Keep pts_times as-is; it aligns 1:1 with files

    # Ensure we always include the first frame as a candidate at t=0
    first_png = os.path.join(candidates_dir, "000000.png")
    if not os.path.exists(first_png):
        cmd_first = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", video_path,
            "-frames:v", "1",
            first_png,
        ]
        code, _, _ = run(cmd_first)
        if code == 0 and os.path.exists(first_png):
            files = [first_png] + files
            pts_times = [0.0] + pts_times
    else:
        # Ensure it is first and time 0.0 is present
        if not files or files[0] != first_png:
            files = [first_png] + files
            pts_times = [0.0] + pts_times

    # Ensure we always include the last frame as a candidate at video end
    # Get duration via ffprobe
    try:
        import json as _json
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=duration", "-of", "json", video_path,
        ]
        code, out, err = run(probe_cmd)
        dur = None
        if code == 0 and out:
            data = _json.loads(out)
            dur = float(data.get("format", {}).get("duration", 0.0))
        if dur is None or dur <= 0:
            dur = pts_times[-1] if pts_times else 0.0
        last_png = os.path.join(candidates_dir, "last.png")
        # Extract a frame very near the end
        cmd_last = [
            "ffmpeg", "-hide_banner", "-y",
            "-sseof", "-0.05",
            "-i", video_path,
            "-frames:v", "1",
            last_png,
        ]
        code, _, _ = run(cmd_last)
        if code == 0 and os.path.exists(last_png):
            files.append(last_png)
            pts_times.append(dur)
    except Exception:
        # Best-effort; ignore if ffprobe/ffmpeg not available
        pass

    return files, pts_times


def sample_dense_frames(video_path: str, out_dir: str, fps: float) -> tuple[list[str], list[float]]:
    """Sample frames at fixed FPS with showinfo timestamps."""
    dense_dir = os.path.join(out_dir, "dense")
    os.makedirs(dense_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "ffmpeg_dense_showinfo.log")
    vf = f"fps={fps},showinfo"
    out_pattern = os.path.join(dense_dir, "%06d.png")
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", video_path,
        "-vf", vf,
        "-vsync", "vfr",
        out_pattern,
    ]
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg dense sampling failed; see log at {log_path}")

    pts_times: list[float] = []
    pts_re = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")
    with open(log_path, "r") as f:
        for line in f:
            m = pts_re.search(line)
            if m:
                try:
                    pts_times.append(float(m.group(1)))
                except ValueError:
                    pass

    files = []
    for i in range(1, len(pts_times) + 1):
        p = os.path.join(dense_dir, f"{i:06d}.png")
        if os.path.exists(p):
            files.append(p)
    return files, pts_times


def dedupe_pages(candidate_files: list[str], candidate_times: list[float], hash_thresh: int, min_gap: float, out_dir: str) -> list[tuple[str, float]]:
    pages_dir = os.path.join(out_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    # Clean previous outputs to avoid stale pages
    for fn in os.listdir(pages_dir):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                os.remove(os.path.join(pages_dir, fn))
            except OSError:
                pass

    kept: list[tuple[str, float]] = []
    last_hash = None
    last_time = -1e9

    for idx, (fp, ts) in enumerate(zip(candidate_files, candidate_times)):
        # Enforce minimum time gap
        if ts - last_time < min_gap:
            continue

        with Image.open(fp) as im:
            im = im.convert("RGB")
            h = imagehash.phash(im)

        if last_hash is not None:
            dist = h - last_hash
            if dist <= hash_thresh:
                # Too similar; skip
                continue

        # Keep: copy into pages with informative name
        label = hhmmss(ts)
        out_name = f"{len(kept)+1:04d}_{label}.png"
        out_path = os.path.join(pages_dir, out_name)
        shutil.copy2(fp, out_path)
        kept.append((out_path, ts))
        last_time = ts
        last_hash = h

    # If nothing kept (edge case), keep the first candidate
    if not kept and candidate_files:
        ts = candidate_times[0]
        label = hhmmss(ts)
        out_name = f"0001_{label}.png"
        out_path = os.path.join(pages_dir, out_name)
        shutil.copy2(candidate_files[0], out_path)
        kept = [(out_path, ts)]

    # Ensure we include the final candidate frame as the last page
    # BUT only if it's not an exact copy of the previously kept page
    if candidate_files:
        last_fp = candidate_files[-1]
        last_ts = candidate_times[-1]
        # Compare on basename presence in kept
        kept_basenames = {os.path.basename(kfp) for kfp, _ in kept}
        if os.path.basename(last_fp) not in kept_basenames:
            is_exact_duplicate = False
            try:
                if kept:
                    # Compare pixel-by-pixel equality with the last kept image
                    with Image.open(kept[-1][0]) as im_prev, Image.open(last_fp) as im_last:
                        im_prev = im_prev.convert("RGB")
                        im_last = im_last.convert("RGB")
                        if im_prev.size == im_last.size:
                            is_exact_duplicate = list(im_prev.getdata()) == list(im_last.getdata())
            except Exception:
                # If comparison fails, fall back to treating as not duplicate
                is_exact_duplicate = False

            if not is_exact_duplicate:
                label = hhmmss(last_ts)
                out_name = f"{len(kept)+1:04d}_{label}.png"
                out_path = os.path.join(pages_dir, out_name)
                try:
                    shutil.copy2(last_fp, out_path)
                    kept.append((out_path, last_ts))
                except OSError:
                    pass

    return kept


def write_index(kept: list[tuple[str, float]], out_dir: str):
    index_csv = os.path.join(out_dir, "index.csv")
    with open(index_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page", "timestamp", "seconds", "file"])
        for i, (fp, ts) in enumerate(kept, start=1):
            w.writerow([i, hhmmss(ts), f"{ts:.3f}", os.path.basename(fp)])
    return index_csv


def build_pdf(kept: list[tuple[str, float]], out_base: str) -> str:
    import img2pdf
    imgs = [fp for fp, _ in kept]
    raw_pdf = out_base + ".raw.pdf"
    with open(raw_pdf, "wb") as f:
        f.write(img2pdf.convert(imgs))

    # OCR pass if available
    ocrmypdf = shutil.which("ocrmypdf")
    final_pdf = out_base + ".pdf"
    if ocrmypdf:
        code, _, err = run([ocrmypdf, "--skip-text", "--optimize", "0", "--jobs", "2", raw_pdf, final_pdf])
        if code != 0:
            # Fallback to non-OCR PDF
            shutil.move(raw_pdf, final_pdf)
        else:
            os.remove(raw_pdf)
    else:
        shutil.move(raw_pdf, final_pdf)
    return final_pdf


def main():
    p = argparse.ArgumentParser(description="Extract slide-like frames from a lecture video and build a searchable PDF.")
    p.add_argument("video", help="Path to input .mp4 video")
    p.add_argument("--outdir", default="slides", help="Base output directory (default: slides)")
    p.add_argument("--scene", type=float, default=0.35, help="FFmpeg scene change threshold (0-1)")
    p.add_argument("--hash", dest="hash_thresh", type=int, default=5, help="Max phash distance to consider images duplicate")
    p.add_argument("--min-gap", dest="min_gap", type=float, default=5.0, help="Minimum seconds between kept slides")
    p.add_argument("--fps", dest="fps", type=float, default=0.0, help="Optional dense sampling FPS (e.g., 0.5 for one frame every 2s)")
    p.add_argument("--max-candidates", dest="max_candidates", type=int, default=0, help="Final cap on kept pages (0 = no cap). First/last always included; uniform downsample.")
    args = p.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"Input not found: {video_path}", file=sys.stderr)
        sys.exit(2)

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(os.path.abspath(args.outdir), base)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] Extracting candidates (scene>{args.scene})...")
    files, times = extract_candidates(video_path, out_dir, args.scene)
    total_sources = [len(files)]
    if args.fps and args.fps > 0:
        print(f"      + dense sampling at {args.fps} fps ...")
        d_files, d_times = sample_dense_frames(video_path, out_dir, args.fps)
        # combine
        combined = list(zip(times, files)) + list(zip(d_times, d_files))
        # de-dup by timestamp+path and sort by time
        combined.sort(key=lambda x: x[0])
        times = [t for t, _ in combined]
        files = [p for _, p in combined]
        total_sources.append(len(d_files))
    print(f"  candidates: {sum(total_sources)} (scene={total_sources[0]} + dense={total_sources[1] if len(total_sources)>1 else 0})")

    print(f"[2/4] Deduplicating with phash<= {args.hash_thresh}, min_gap={args.min_gap}s ...")
    kept = dedupe_pages(files, times, args.hash_thresh, args.min_gap, out_dir)
    # Apply final cap on candidates if requested (keep first and last; uniform spread)
    if args.max_candidates and args.max_candidates > 0 and len(kept) > args.max_candidates:
        n = len(kept)
        m = args.max_candidates
        sel = set()
        if m == 1:
            sel.add(0)
        else:
            for k in range(m):
                idx = round(k * (n - 1) / (m - 1))
                sel.add(int(idx))
        idxs = sorted(sel)
        kept_idxs = [kept[i] for i in idxs]

        # Clean unselected page images from pages/ directory
        pages_dir = os.path.join(out_dir, "pages")
        keep_basenames = {os.path.basename(fp) for fp, _ in kept_idxs}
        for fn in os.listdir(pages_dir):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")) and fn not in keep_basenames:
                try:
                    os.remove(os.path.join(pages_dir, fn))
                except OSError:
                    pass
        kept = kept_idxs
    print(f"  kept pages: {len(kept)}")

    print(f"[3/4] Writing index.csv ...")
    index_csv = write_index(kept, out_dir)
    print(f"  index: {index_csv}")

    print(f"[4/4] Building searchable PDF ...")
    out_base = os.path.join(os.path.abspath(args.outdir), base)
    final_pdf = build_pdf(kept, out_base)
    print(f"  pdf: {final_pdf}")

    print("Done.")


if __name__ == "__main__":
    main()
