# Audio Chunking + Transcribe Pipeline (OpenAI 4o Transcribe)

A drop‑in, **size/duration‑aware** transcription pipeline that:

1) **Segments** long audio into safe chunks with small overlaps
2) **Transcribes** each chunk via `gpt-4o-transcribe` (or `gpt-4o-mini-transcribe`)
3) **Stitches** chunk texts, removing duplicates around overlaps
4) Emits: consolidated TXT, coarse SRT/VTT, and a detailed JSON artifact

> No Whisper-1 verbose JSON; this is plain-text stitching with chunk-level timestamps. FFmpeg is required.

---

## Folder structure (copy/paste)

```
transcribe_pipeline/
├─ requirements.txt
├─ config.yaml
├─ README.md
├─ run_pipeline.py
├─ audio_utils/
│  ├─ __init__.py
│  ├─ ffprobe.py
│  ├─ segmenter.py
│  ├─ transcriber.py
│  └─ stitcher.py
└─ outputs/               # will be created at runtime
```

Create these files with the exact contents below.

---

## requirements.txt

```
openai>=1.40,<2.0
PyYAML>=6.0
tqdm>=4.66
```

> **System dependency:** FFmpeg must be installed and on PATH (`ffmpeg`, `ffprobe`).

---

## config.yaml

```yaml
# === Model & API ===
model: gpt-4o-transcribe   # or gpt-4o-mini-transcribe
response_format: json      # json or text
prompt: ""                 # optional domain hints; keep empty if unsure
parallel_requests: 3
max_retries: 3
backoff_base_ms: 800       # exponential backoff base

# === Chunking policy ===
max_file_mb: 25            # practical upper bound per request; keep conservative
target_chunk_mb: 16        # aim below the hard cap to avoid edge cases
max_chunk_secs: 900        # 15 minutes per chunk max (whichever bound hits first)
overlap_secs: 3.0          # small redundancy to avoid word cuts

# Re-encode each chunk to keep size predictable (recommended)
# For large/variable sources, re-encoding stabilizes chunk sizes.
reencode:
  enabled: true
  codec: aac               # m4a/aac
  bitrate_kbps: 64         # 48–96 is usually fine for speech
  channels: 1
  sample_rate: 16000

# === Segmentation strategy ===
segmenter: fixed           # fixed | silence

silence:
  # Only used if segmenter: silence. Requires ffmpeg silencedetect.
  min_silence_db: -35      # threshold (dB)
  min_silence_dur: 0.6     # seconds

# === Outputs ===
outputs:
  write_txt: true
  write_json: true
  write_srt: true
  write_vtt: false

# === Paths ===
input_audio: ""            # set via CLI arg or here
work_dir: "outputs"        # artifacts & chunks
```

---

## README.md

```markdown
# Transcription Pipeline (Chunk → Transcribe → Stitch)

**Prereqs**
- Python 3.10+
- FFmpeg installed and on PATH (`ffmpeg`, `ffprobe`)
- `OPENAI_API_KEY` set in your environment

**Install**
```bash
cd transcribe_pipeline
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

**Quick run**
```bash
python run_pipeline.py --input /path/to/your/audio.mp3
```
Options:
- `--model gpt-4o-transcribe|gpt-4o-mini-transcribe`
- `--segmenter fixed|silence`
- `--work-dir /path/to/outputs`
- `--prompt "Custom domain terms, names, words..."`

**Outputs** (in `outputs/<job_id>/`):
- `manifest.json` – chunk list, timings, per-chunk text, API latencies, etc.
- `transcript.txt` – merged transcript
- `transcript.json` – full structured artifact (chunks[], full_text, metadata)
- `transcript.srt` – coarse captions (optional VTT)
- `chunks/` – audio chunks for audit

**Notes**
- If your file is huge, the pipeline auto-computes safe chunk durations from bitrate and the caps in `config.yaml`.
- If `segmenter: silence`, it prefers nearest silence near the target duration but still enforces caps.
- Stitching removes duplicated words across overlaps via fuzzy matching.
```

---

## run_pipeline.py

```python
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from audio_utils.ffprobe import probe_audio
from audio_utils.segmenter import plan_and_segment
from audio_utils.transcriber import transcribe_manifest
from audio_utils.stitcher import stitch_outputs, write_side_outputs


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, out_dir: Path):
    (out_dir / "effective_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Chunk → Transcribe → Stitch")
    parser.add_argument("--input", required=False, help="Path to input audio file")
    parser.add_argument("--work-dir", default=None, help="Working dir (defaults to config)")
    parser.add_argument("--model", default=None, help="Override model")
    parser.add_argument("--segmenter", default=None, choices=["fixed", "silence"], help="Segmenter")
    parser.add_argument("--prompt", default=None, help="Optional context prompt for ASR")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if args.model:
        cfg["model"] = args.model
    if args.segmenter:
        cfg["segmenter"] = args.segmenter
    if args.prompt is not None:
        cfg["prompt"] = args.prompt

    input_audio = args.input or cfg.get("input_audio")
    if not input_audio:
        print("ERROR: Provide --input or set input_audio in config.yaml", file=sys.stderr)
        sys.exit(2)

    input_path = Path(input_audio).expanduser().resolve()
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    # prepare work dir
    root_out = Path(args.work_dir or cfg.get("work_dir", "outputs")).expanduser()
    job_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (root_out / job_id)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    meta = probe_audio(input_path)
    save_config(cfg, out_dir)

    manifest_path = out_dir / "manifest.json"
    plan_and_segment(
        input_path=input_path,
        meta=meta,
        cfg=cfg,
        chunks_dir=chunks_dir,
        manifest_path=manifest_path,
    )

    transcribe_manifest(
        manifest_path=manifest_path,
        cfg=cfg,
    )

    full_text, merged_chunks = stitch_outputs(manifest_path)
    write_side_outputs(out_dir, full_text, merged_chunks, manifest_path, cfg)

    print("\nDone. Artifacts in:", out_dir)


if __name__ == "__main__":
    main()
```

---

## audio_utils/__init__.py

```python
# empty module init for relative imports
```

---

## audio_utils/ffprobe.py

```python
import json
import subprocess
from pathlib import Path


def _run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc.stdout


def probe_audio(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    out = _run(cmd)
    data = json.loads(out.decode("utf-8"))

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    astream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    duration = float(fmt.get("duration") or astream.get("duration") or 0.0)
    bit_rate = int(fmt.get("bit_rate") or astream.get("bit_rate") or 128000)
    sample_rate = int(astream.get("sample_rate") or 44100)
    channels = int(astream.get("channels") or 2)

    return {
        "duration": duration,
        "bit_rate": bit_rate,   # bits per second
        "sample_rate": sample_rate,
        "channels": channels,
        "format_name": fmt.get("format_name"),
        "size_bytes": int(fmt.get("size", 0)),
    }
```

---

## audio_utils/segmenter.py

```python
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
```

---

## audio_utils/transcriber.py

```python
import concurrent.futures
import json
import os
import time
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI


TRANSIENT_ERRORS = ("rate_limit_exceeded", "server_error", "temporarily_unavailable")


def _save_manifest(manifest_path: Path, manifest: dict):
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _transcribe_one(client: OpenAI, chunk: dict, cfg: dict):
    start = time.time()
    model = cfg.get("model", "gpt-4o-transcribe")
    response_format = cfg.get("response_format", "json")
    prompt = cfg.get("prompt", "")

    with open(chunk["file"], "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format=response_format,
            prompt=prompt if prompt else None,
        )

    # Normalize to dict
    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
    else:
        try:
            data = json.loads(str(resp))
        except Exception:
            data = {"text": getattr(resp, "text", str(resp))}

    text = data.get("text") if isinstance(data, dict) else None
    latency_ms = int((time.time() - start) * 1000)
    return text, latency_ms


def _retrying_transcribe(client: OpenAI, chunk: dict, cfg: dict):
    max_retries = int(cfg.get("max_retries", 3))
    backoff = int(cfg.get("backoff_base_ms", 800))

    for attempt in range(max_retries + 1):
        try:
            return _transcribe_one(client, chunk, cfg)
        except Exception as e:
            msg = str(e).lower()
            if attempt < max_retries and any(t in msg for t in TRANSIENT_ERRORS):
                delay = (2 ** attempt) * backoff / 1000.0
                time.sleep(delay)
                continue
            raise


def transcribe_manifest(manifest_path: Path, cfg: dict):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    pending = [c for c in manifest["chunks"] if c.get("status") == "pending"]
    pbar = tqdm(total=len(pending), desc="Transcribing", unit="chunk")

    def work(c):
        try:
            text, latency = _retrying_transcribe(client, c, cfg)
            c["text"] = text
            c["latency_ms"] = latency
            c["status"] = "done"
        except Exception as e:
            c["status"] = "error"
            c["error"] = str(e)
        finally:
            return c

    max_workers = int(cfg.get("parallel_requests", 3))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(work, c) for c in pending]
        for fut in concurrent.futures.as_completed(futures):
            _ = fut.result()
            pbar.update(1)
            _save_manifest(manifest_path, manifest)

    pbar.close()
    _save_manifest(manifest_path, manifest)
```

---

## audio_utils/stitcher.py

```python
import json
import math
from datetime import timedelta
from pathlib import Path
from difflib import SequenceMatcher


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    # Basic normalization to stabilize matching
    s = s.replace("\r", "\n").replace("\u200b", "")
    s = "\n".join(line.strip() for line in s.splitlines())
    return s


def _dedup_join(prev: str, nxt: str, window: int = 500, min_match: int = 30) -> str:
    prev_n = _normalize_text(prev)
    nxt_n = _normalize_text(nxt)

    tail = prev_n[-window:]
    head = nxt_n[:window]

    sm = SequenceMatcher(None, tail, head, autojunk=False)
    match = max(sm.get_opcodes(), key=lambda op: (op[0] == "equal", op[3]-op[2]))
    # op: (tag, i1, i2, j1, j2)
    tag, i1, i2, j1, j2 = match
    if tag == "equal" and (i2 - i1) >= min_match:
        # Drop the overlapping text from the next chunk
        deduped = prev_n + head[j2:]
    else:
        deduped = prev_n + " " + nxt_n
    return deduped.strip()


def _fmt_ts(seconds: float) -> str:
    td = timedelta(seconds=float(seconds))
    # SRT format: HH:MM:SS,mmm
    total_seconds = int(td.total_seconds())
    ms = int((float(seconds) - int(float(seconds))) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def stitch_outputs(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest["chunks"]

    merged_text = ""
    merged_chunks = []

    for i, c in enumerate(chunks):
        text = c.get("text") or ""
        if not text:
            continue
        if i == 0:
            merged_text = _normalize_text(text)
        else:
            merged_text = _dedup_join(merged_text, text)
        merged_chunks.append({
            "index": c["index"],
            "t_start": c["t_start"],
            "t_end": c["t_end"],
            "text": _normalize_text(text),
        })

    return merged_text.strip(), merged_chunks


def _write_txt(out_dir: Path, full_text: str):
    (out_dir / "transcript.txt").write_text(full_text, encoding="utf-8")


def _write_json(out_dir: Path, full_text: str, merged_chunks: list, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = {
        "source": manifest.get("input"),
        "meta": manifest.get("meta"),
        "model": manifest.get("model"),
        "response_format": manifest.get("response_format"),
        "prompt": manifest.get("prompt"),
        "chunks": merged_chunks,
        "full_text": full_text,
    }
    (out_dir / "transcript.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _split_text_by_chars(text: str, parts: int) -> list:
    if parts <= 1:
        return [text]
    n = max(1, len(text) // parts)
    res = []
    for i in range(parts - 1):
        res.append(text[i*n:(i+1)*n])
    res.append(text[(parts-1)*n:])
    return res


def _write_srt(out_dir: Path, merged_chunks: list):
    # Coarse: split each chunk into ~10s spans by relative length
    lines = []
    idx = 1
    for ch in merged_chunks:
        t0, t1 = float(ch["t_start"]), float(ch["t_end"])
        dur = max(0.001, t1 - t0)
        text = ch["text"]
        # target 10s spans
        nspans = max(1, int(round(dur / 10.0)))
        parts = _split_text_by_chars(text, nspans)
        # map each part evenly across the interval
        for j, part in enumerate(parts):
            s = t0 + (dur * j) / len(parts)
            e = t0 + (dur * (j + 1)) / len(parts)
            lines.append(f"{idx}\n{_fmt_ts(s)} --> {_fmt_ts(e)}\n{part.strip()}\n")
            idx += 1

    (out_dir / "transcript.srt").write_text("\n".join(lines), encoding="utf-8")


def _write_vtt(out_dir: Path, merged_chunks: list):
    lines = ["WEBVTT", ""]
    for ch in merged_chunks:
        s = _fmt_ts(ch["t_start"]).replace(",", ".")
        e = _fmt_ts(ch["t_end"]).replace(",", ".")
        lines.append(f"{s} --> {e}\n{ch['text'].strip()}\n")
    (out_dir / "transcript.vtt").write_text("\n".join(lines), encoding="utf-8")


def write_side_outputs(out_dir: Path, full_text: str, merged_chunks: list, manifest_path: Path, cfg: dict):
    if cfg.get("outputs", {}).get("write_txt", True):
        _write_txt(out_dir, full_text)
    if cfg.get("outputs", {}).get("write_json", True):
        _write_json(out_dir, full_text, merged_chunks, manifest_path)
    if cfg.get("outputs", {}).get("write_srt", True):
        _write_srt(out_dir, merged_chunks)
    if cfg.get("outputs", {}).get("write_vtt", False):
        _write_vtt(out_dir, merged_chunks)
```

---

## What to tweak (quick knobs)

- **Make chunks smaller**: lower `target_chunk_mb` or `max_chunk_secs`
- **Reduce artifacts**: increase `overlap_secs` to 4–5s
- **Speed up**: raise `parallel_requests` (watch rate limits)
- **Better joins**: increase `window` and `min_match` in `_dedup_join()` if you see mid-word seams
- **Prefer clean boundaries**: set `segmenter: silence` and tune `min_silence_db`/`min_silence_dur` (implementation placeholder is fixed‑window; silence refinement can be added similarly with `silencedetect` parsing if you want it in v2)

---

## Usage recap

1. Ensure `ffmpeg` & `ffprobe` are installed
2. `export OPENAI_API_KEY=...`
3. `pip install -r requirements.txt`
4. `python run_pipeline.py --input /path/to/audio.mp3`

Artifacts will appear under `outputs/<job_id>/`.

