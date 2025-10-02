"""
Pytest configuration and shared fixtures for the transcribe pipeline tests.

This module provides common test fixtures and configuration that can be
used across all test modules.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

from src.transcribe_pipeline.dependencies.mocks import (
    MockSubprocessProvider,
    MockOpenAIClientProvider,
    MockFileSystemProvider,
    MockTimeProvider,
    MockEnvironmentProvider,
)
from src.transcribe_pipeline.config.pipeline_config import create_default_config


@pytest.fixture
def mock_subprocess_provider():
    """Provide a mock subprocess provider for testing."""
    return MockSubprocessProvider()


@pytest.fixture
def mock_openai_provider():
    """Provide a mock OpenAI client provider for testing."""
    return MockOpenAIClientProvider()


@pytest.fixture
def mock_fs_provider():
    """Provide a mock file system provider for testing."""
    return MockFileSystemProvider()


@pytest.fixture
def mock_time_provider():
    """Provide a mock time provider for testing."""
    return MockTimeProvider()


@pytest.fixture
def mock_env_provider():
    """Provide a mock environment provider for testing."""
    provider = MockEnvironmentProvider()
    provider.set_variable("OPENAI_API_KEY", "test-api-key")
    return provider


@pytest.fixture
def sample_config():
    """Provide a default configuration for testing."""
    config = create_default_config()
    # Override with test-friendly settings
    config.model.max_retries = 1
    config.model.backoff_base_ms = 100
    config.model.parallel_requests = 1
    config.chunking.target_chunk_mb = 1  # Small chunks for testing
    config.chunking.max_chunk_secs = 60
    config.chunking.overlap_secs = 1.0
    return config


@pytest.fixture
def sample_audio_metadata():
    """Provide sample audio metadata for testing."""
    return {
        "duration": 120.5,  # 2 minutes
        "bit_rate": 128000,  # 128 kbps
        "sample_rate": 44100,
        "channels": 2,
        "format_name": "mp3",
        "size_bytes": 1920000,  # ~1.9MB
    }


@pytest.fixture
def sample_manifest():
    """Provide a sample manifest for testing."""
    return {
        "input": "/path/to/audio.mp3",
        "meta": {
            "duration": 120.5,
            "bit_rate": 128000,
            "sample_rate": 44100,
            "channels": 2,
            "format_name": "mp3",
            "size_bytes": 1920000,
        },
        "cfg_hash": None,
        "chunks": [
            {
                "index": 0,
                "file": "/path/to/chunks/chunk_0000.m4a",
                "t_start": 0.0,
                "t_end": 60.0,
                "overlap_head": 0.0,
                "overlap_tail": 1.0,
                "status": "pending",
                "text": None,
                "latency_ms": None,
                "retries": 0,
            },
            {
                "index": 1,
                "file": "/path/to/chunks/chunk_0001.m4a",
                "t_start": 59.0,
                "t_end": 120.5,
                "overlap_head": 1.0,
                "overlap_tail": 0.0,
                "status": "pending",
                "text": None,
                "latency_ms": None,
                "retries": 0,
            },
        ],
        "model": "gpt-4o-transcribe",
        "response_format": "json",
        "prompt": "",
    }


@pytest.fixture
def sample_transcription_response():
    """Provide a sample OpenAI transcription response."""
    return {
        "text": "This is a sample transcription of the audio content.",
        "language": "en",
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_audio_file(temp_dir):
    """Provide a sample audio file path for testing."""
    audio_file = temp_dir / "sample.mp3"
    # Create a dummy file
    audio_file.write_bytes(b"dummy audio content")
    return audio_file


@pytest.fixture
def sample_chunk_file(temp_dir):
    """Provide a sample chunk file path for testing."""
    chunk_file = temp_dir / "chunk_0000.m4a"
    # Create a dummy chunk file
    chunk_file.write_bytes(b"dummy chunk content")
    return chunk_file


@pytest.fixture
def mock_ffprobe_output():
    """Provide mock FFprobe JSON output for testing."""
    return json.dumps({
        "format": {
            "duration": "120.5",
            "bit_rate": "128000",
            "format_name": "mp3",
            "size": "1920000"
        },
        "streams": [
            {
                "codec_type": "audio",
                "sample_rate": "44100",
                "channels": 2
            }
        ]
    }).encode('utf-8')


@pytest.fixture
def mock_ffprobe_success_result(mock_ffprobe_output):
    """Provide a mock successful FFprobe result."""
    import subprocess
    return subprocess.CompletedProcess(
        args=["ffprobe", "test.mp3"],
        returncode=0,
        stdout=mock_ffprobe_output,
        stderr=b""
    )


@pytest.fixture
def mock_ffmpeg_success_result():
    """Provide a mock successful FFmpeg result."""
    import subprocess
    return subprocess.CompletedProcess(
        args=["ffmpeg", "test.mp3"],
        returncode=0,
        stdout=b"",
        stderr=b""
    )
