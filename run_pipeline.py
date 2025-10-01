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