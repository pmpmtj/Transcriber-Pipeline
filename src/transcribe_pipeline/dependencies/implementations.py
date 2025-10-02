"""
Production Dependency Implementations

This module provides real implementations of the dependency interfaces
for use in production code.
"""

import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from openai import OpenAI

from .interfaces import (
    SubprocessProvider,
    OpenAIClientProvider,
    FileSystemProvider,
    TimeProvider,
    EnvironmentProvider,
)


class RealSubprocessProvider(SubprocessProvider):
    """Real implementation of subprocess operations."""
    
    def run(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a subprocess command with real subprocess."""
        return subprocess.run(cmd, **kwargs)


class RealOpenAIClientProvider(OpenAIClientProvider):
    """Real implementation of OpenAI API operations."""
    
    def create_client(self, api_key: str) -> Any:
        """Create a real OpenAI client instance."""
        return OpenAI(api_key=api_key)
    
    def transcribe_audio(self, client: Any, audio_file: Path, model: str, 
                        response_format: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio using real OpenAI API."""
        with open(audio_file, "rb") as f:
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
            import json
            try:
                data = json.loads(str(resp))
            except Exception:
                data = {"text": getattr(resp, "text", str(resp))}
        
        return data


class RealFileSystemProvider(FileSystemProvider):
    """Real implementation of file system operations."""
    
    def read_text(self, path: Path, encoding: str = "utf-8") -> str:
        """Read text from a real file."""
        return path.read_text(encoding=encoding)
    
    def write_text(self, path: Path, content: str, encoding: str = "utf-8") -> None:
        """Write text to a real file."""
        path.write_text(content, encoding=encoding)
    
    def read_binary(self, path: Path) -> bytes:
        """Read binary data from a real file."""
        return path.read_bytes()
    
    def exists(self, path: Path) -> bool:
        """Check if a real path exists."""
        return path.exists()
    
    def mkdir(self, path: Path, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a real directory."""
        path.mkdir(parents=parents, exist_ok=exist_ok)


class RealTimeProvider(TimeProvider):
    """Real implementation of time operations."""
    
    def time(self) -> float:
        """Get real current time."""
        return time.time()
    
    def sleep(self, seconds: float) -> None:
        """Real sleep operation."""
        time.sleep(seconds)
    
    def strftime(self, format_str: str) -> str:
        """Format real current time."""
        return time.strftime(format_str)


class RealEnvironmentProvider(EnvironmentProvider):
    """Real implementation of environment variable access."""
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get real environment variable value."""
        return os.getenv(key, default)
    
    def get_required(self, key: str) -> str:
        """Get required real environment variable value."""
        value = os.getenv(key)
        if not value:
            raise RuntimeError(f"{key} is not set")
        return value
