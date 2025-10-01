import json
import math
import subprocess
from pathlib import Path


def _run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc


def _safe_chunk_window(meta, cfg):
    # Compute chunk seconds from target MB & bitrate; guard with max_chunk_secs
    # MB ≈ (kbps * secs) / (8*1000)  => secs ≈ MB * 8000 / kbps
    bitrate_kbps = max(1, int(meta["bit_rate"]) // 1000)
    t_by_mb = cfg["target_chunk_mb"] * 8000 / bitrate_kbps
    t_max = cfg.get("max_chunk_secs", 900)
    chunk_secs = max(60, min(t_by_mb, t_max))  # at least 60s
    return float(chunk_secs)


def _encode_args(cfg):
    if not cfg.get("reencode", {}).get("enabled", True):
        return []
    r = cfg["reencode"]
    return [
        "-ac", str(r.get("channels", 1)),
        "-ar", str(r.get("sample_rate", 16000)),
        "-b:a", f"{r.get('bitrate_kbps', 64)}k",
        "-c:a", r.get("codec", "aac"),
    ]


def _out_ext(cfg):
    if not cfg.get("reencode", {}).get("enabled", True):
        return ".m4a"  # container copy fallback
    codec = cfg["reencode"].get("codec", "aac").lower()
    return ".m4a" if codec in ("aac", "libfdk_aac") else ".wav"


def _format_time(seconds: float) -> str:
    # HH:MM:SS.mmm
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def plan_and_segment(input_path: Path, meta: dict, cfg: dict, chunks_dir: Path, manifest_path: Path):
    duration = float(meta["duration"]) or 0.0
    if duration <= 0.0:
        raise ValueError("Could not determine audio duration.")

    overlap = float(cfg.get("overlap_secs", 3.0))
    window = _safe_chunk_window(meta, cfg)

    # Simple fixed windows (optional: add silence-aware refinement later)
    cut_points = []
    t = 0.0
    while t < duration:
        start = max(0.0, t - (overlap if t > 0 else 0.0))
        end = min(duration, t + window + (overlap if t + window < duration else 0.0))
        cut_points.append((start, end))
        t += window

    encode_args = _encode_args(cfg)
    ext = _out_ext(cfg)

    chunks = []
    for idx, (start, end) in enumerate(cut_points):
        out_name = f"chunk_{idx:04d}{ext}"
        out_path = chunks_dir / out_name
        length = max(0.01, end - start)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(input_path),
            "-t", f"{length:.3f}",
        ] + encode_args + [str(out_path)]
        _run(cmd)

        chunks.append({
            "index": idx,
            "file": str(out_path),
            "t_start": start,
            "t_end": end,
            "overlap_head": overlap if idx > 0 else 0.0,
            "overlap_tail": overlap if end < duration else 0.0,
            "status": "pending",
            "text": None,
            "latency_ms": None,
            "retries": 0,
        })

    manifest = {
        "input": str(input_path),
        "meta": meta,
        "cfg_hash": None,  # optional: inject a hash of cfg
        "chunks": chunks,
        "model": cfg.get("model"),
        "response_format": cfg.get("response_format", "json"),
        "prompt": cfg.get("prompt", ""),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")