import json
from pathlib import Path
from typing import Dict, Any, Optional

from ..dependencies.interfaces import SubprocessProvider
from ..dependencies.implementations import RealSubprocessProvider


def _run(cmd, subprocess_provider: SubprocessProvider):
    """Run FFprobe command using the provided subprocess provider."""
    import subprocess
    proc = subprocess_provider.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc.stdout


def probe_audio(
    path: Path, 
    subprocess_provider: Optional[SubprocessProvider] = None
) -> Dict[str, Any]:
    """Probe audio file metadata using FFprobe with dependency injection."""
    # Use default provider if not provided (for backward compatibility)
    if subprocess_provider is None:
        subprocess_provider = RealSubprocessProvider()
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    out = _run(cmd, subprocess_provider)
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