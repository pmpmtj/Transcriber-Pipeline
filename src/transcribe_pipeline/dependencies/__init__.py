"""
Dependency Injection Framework for Transcribe Pipeline

This module provides a dependency injection framework to make the transcription
pipeline more testable by abstracting external dependencies such as:
- Subprocess operations (FFmpeg)
- OpenAI API calls
- File system operations
- Time operations
- Environment variables

The framework consists of:
- interfaces.py: Abstract base classes defining contracts
- implementations.py: Production implementations
- mocks.py: Mock implementations for testing
"""

from .interfaces import (
    SubprocessProvider,
    OpenAIClientProvider,
    FileSystemProvider,
    TimeProvider,
    EnvironmentProvider,
)
from .implementations import (
    RealSubprocessProvider,
    RealOpenAIClientProvider,
    RealFileSystemProvider,
    RealTimeProvider,
    RealEnvironmentProvider,
)
from .mocks import (
    MockSubprocessProvider,
    MockOpenAIClientProvider,
    MockFileSystemProvider,
    MockTimeProvider,
    MockEnvironmentProvider,
)

__all__ = [
    # Interfaces
    "SubprocessProvider",
    "OpenAIClientProvider", 
    "FileSystemProvider",
    "TimeProvider",
    "EnvironmentProvider",
    # Production implementations
    "RealSubprocessProvider",
    "RealOpenAIClientProvider",
    "RealFileSystemProvider", 
    "RealTimeProvider",
    "RealEnvironmentProvider",
    # Mock implementations
    "MockSubprocessProvider",
    "MockOpenAIClientProvider",
    "MockFileSystemProvider",
    "MockTimeProvider", 
    "MockEnvironmentProvider",
]
