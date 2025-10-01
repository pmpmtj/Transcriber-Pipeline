# Transcription Pipeline (Chunk → Transcribe → Stitch)


**Prereqs**
- Python 3.10+
- FFmpeg installed and on PATH (`ffmpeg`, `ffprobe`)
- `OPENAI_API_KEY` set in your environment


**Install**
```bash
cd transcribe_pipeline
python -m venv .venv && source .venv/bin/activate # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt