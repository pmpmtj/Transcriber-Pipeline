# Transcribe Pipeline

A robust audio transcription pipeline that chunks, transcribes, and stitches audio files using OpenAI's Whisper API with configurable settings and professional logging.

## Features

- **Intelligent Chunking**: Automatically segments large audio files into optimal chunks
- **OpenAI Whisper Integration**: Uses GPT-4o-transcribe or GPT-4o-mini-transcribe models
- **Multiple Output Formats**: Supports TXT, JSON, SRT, and VTT formats
- **Configurable Settings**: Class-based configuration with validation
- **Professional Logging**: Integrated logging system with debug information
- **CLI Interface**: Simple command-line interface for easy usage

## Prerequisites

- Python 3.10+
- FFmpeg installed and on PATH (`ffmpeg`, `ffprobe`)
- `OPENAI_API_KEY` set in your environment

## Installation

### From Source (Development)
```bash
git clone <repository-url>
cd transcribe_pipeline
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
pip install -e .
```

### From PyPI (Future)
```bash
pip install transcribe_pipeline
```

## Usage

### Command Line Interface

```bash
# Basic usage
transcribe-pipeline audio.mp3

# With custom model
transcribe-pipeline audio.mp3 --model gpt-4o-mini-transcribe

# With custom segmentation and prompt
transcribe-pipeline audio.mp3 --segmenter silence --prompt "Technical presentation"

# With custom output directory
transcribe-pipeline audio.mp3 --work-dir ./transcripts
```

### Python API

```python
from transcribe_pipeline import PipelineConfig, create_default_config

# Create configuration
config = create_default_config()
config.model.model = "gpt-4o-mini-transcribe"
config.chunking.target_chunk_mb = 20

# Use in your code...
```

## Output Files

The pipeline generates the following files in the output directory:
- `transcript.txt` - Plain text transcription
- `transcript.json` - Structured JSON with metadata
- `transcript.srt` - SubRip subtitle format
- `transcript.vtt` - WebVTT format (if enabled)
- `effective_config.json` - Configuration used for the run
- `manifest.json` - Processing manifest with chunk information