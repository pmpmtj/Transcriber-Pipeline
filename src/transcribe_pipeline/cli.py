"""
Command Line Interface for Transcribe Pipeline

Simplified CLI that takes audio file as positional argument and supports
optional configuration overrides.
"""

import argparse
import sys
import time
from pathlib import Path

from .config.pipeline_config import create_default_config
from .audio_utils.ffprobe import probe_audio
from .audio_utils.segmenter import plan_and_segment
from .audio_utils.transcriber import transcribe_manifest
from .audio_utils.stitcher import stitch_outputs, write_side_outputs


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI's Whisper API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  transcribe-pipeline audio.mp3
  transcribe-pipeline audio.mp3 --model gpt-4o-mini-transcribe
  transcribe-pipeline audio.mp3 --segmenter silence --prompt "Technical presentation"
  transcribe-pipeline audio.mp3 --work-dir ./transcripts
        """
    )
    
    # Required positional argument for input audio file
    parser.add_argument(
        "audio_file",
        help="Path to input audio file"
    )
    
    # Optional configuration overrides
    parser.add_argument(
        "--work-dir", 
        default=None, 
        help="Working directory for outputs (default: outputs/)"
    )
    parser.add_argument(
        "--model", 
        default=None, 
        choices=["gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
        help="OpenAI model to use (default: gpt-4o-transcribe)"
    )
    parser.add_argument(
        "--segmenter", 
        default=None, 
        choices=["fixed", "silence"], 
        help="Segmentation strategy (default: fixed)"
    )
    parser.add_argument(
        "--prompt", 
        default=None, 
        help="Optional context prompt for better transcription accuracy"
    )
    
    args = parser.parse_args()

    try:
        # Create configuration instance
        config = create_default_config()
        
        # Apply CLI overrides
        config.apply_cli_overrides(args)
        
        # Get input audio path
        input_path = config.get_input_audio_path(args.audio_file)
        
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

        print(f"\n[SUCCESS] Transcription completed successfully!")
        print(f"[INFO] Output directory: {out_dir}")
        print(f"[INFO] Files generated:")
        if config.outputs.write_txt:
            print(f"   - transcript.txt")
        if config.outputs.write_json:
            print(f"   - transcript.json")
        if config.outputs.write_srt:
            print(f"   - transcript.srt")
        if config.outputs.write_vtt:
            print(f"   - transcript.vtt")

    except KeyboardInterrupt:
        print("\n[ERROR] Transcription cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
