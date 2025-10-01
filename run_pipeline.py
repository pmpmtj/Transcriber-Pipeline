import argparse
import json
import os
import sys
import time
from pathlib import Path

from config.pipeline_config import PipelineConfig, create_default_config
from audio_utils.ffprobe import probe_audio
from audio_utils.segmenter import plan_and_segment
from audio_utils.transcriber import transcribe_manifest
from audio_utils.stitcher import stitch_outputs, write_side_outputs


def main():
    parser = argparse.ArgumentParser(description="Chunk -> Transcribe -> Stitch")
    parser.add_argument("--input", required=False, help="Path to input audio file")
    parser.add_argument("--work-dir", default=None, help="Working dir (defaults to config)")
    parser.add_argument("--model", default=None, help="Override model")
    parser.add_argument("--segmenter", default=None, choices=["fixed", "silence"], help="Segmenter")
    parser.add_argument("--prompt", default=None, help="Optional context prompt for ASR")
    args = parser.parse_args()

    # Create configuration instance
    config = create_default_config()
    
    # Apply CLI overrides
    config.apply_cli_overrides(args)
    
    # Get input audio path
    try:
        input_path = config.get_input_audio_path(args.input)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # Prepare work directory
    root_out = config.get_work_directory(args.work_dir)
    job_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = root_out / job_id
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration for debugging
    config.log_configuration()

    # Probe audio and save effective configuration
    meta = probe_audio(input_path)
    config.save_effective_config(out_dir)

    # Run the pipeline
    manifest_path = out_dir / "manifest.json"
    plan_and_segment(
        input_path=input_path,
        meta=meta,
        config=config,
        chunks_dir=chunks_dir,
        manifest_path=manifest_path,
    )

    transcribe_manifest(
        manifest_path=manifest_path,
        config=config,
    )

    full_text, merged_chunks = stitch_outputs(manifest_path)
    write_side_outputs(out_dir, full_text, merged_chunks, manifest_path, config)

    print("\nDone. Artifacts in:", out_dir)


if __name__ == "__main__":
    main()