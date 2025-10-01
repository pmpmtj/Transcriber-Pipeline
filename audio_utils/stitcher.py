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


def write_side_outputs(out_dir: Path, full_text: str, merged_chunks: list, manifest_path: Path, config):
    if config.outputs.write_txt:
        _write_txt(out_dir, full_text)
    if config.outputs.write_json:
        _write_json(out_dir, full_text, merged_chunks, manifest_path)
    if config.outputs.write_srt:
        _write_srt(out_dir, merged_chunks)
    if config.outputs.write_vtt:
        _write_vtt(out_dir, merged_chunks)