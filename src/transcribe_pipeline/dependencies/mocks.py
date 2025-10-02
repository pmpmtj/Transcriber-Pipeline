"""
Mock Dependency Implementations

This module provides mock implementations of the dependency interfaces
for use in unit testing. These mocks allow for controlled testing
without external dependencies.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock

from .interfaces import (
    SubprocessProvider,
    OpenAIClientProvider,
    FileSystemProvider,
    TimeProvider,
    EnvironmentProvider,
)


class MockSubprocessProvider(SubprocessProvider):
    """Mock implementation of subprocess operations for testing."""
    
    def __init__(self):
        self.commands_run = []
        self.results = {}  # cmd_str -> result mapping
        self.default_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"", stderr=b""
        )
    
    def set_result(self, cmd: List[str], result: subprocess.CompletedProcess):
        """Set expected result for a command."""
        cmd_str = " ".join(cmd)
        self.results[cmd_str] = result
    
    def run(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a mock subprocess command."""
        cmd_str = " ".join(cmd)
        self.commands_run.append((cmd, kwargs))
        
        if cmd_str in self.results:
            return self.results[cmd_str]
        
        return self.default_result


class MockOpenAIClientProvider(OpenAIClientProvider):
    """Mock implementation of OpenAI API operations for testing."""
    
    def __init__(self):
        self.clients_created = []
        self.transcriptions = {}  # file_path -> result mapping
        self.default_transcription = {"text": "Mock transcription text"}
        self.transcription_errors = {}  # file_path -> exception mapping
    
    def set_transcription_result(self, audio_file: Path, result: Dict[str, Any]):
        """Set expected transcription result for an audio file."""
        self.transcriptions[str(audio_file)] = result
    
    def set_transcription_error(self, audio_file: Path, exception: Exception):
        """Set expected transcription error for an audio file."""
        self.transcription_errors[str(audio_file)] = exception
    
    def create_client(self, api_key: str) -> Any:
        """Create a mock OpenAI client."""
        mock_client = Mock()
        mock_client.api_key = api_key
        self.clients_created.append((api_key, mock_client))
        return mock_client
    
    def transcribe_audio(self, client: Any, audio_file: Path, model: str, 
                        response_format: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Mock transcription operation."""
        file_key = str(audio_file)
        
        # Check for expected error
        if file_key in self.transcription_errors:
            raise self.transcription_errors[file_key]
        
        # Return expected result or default
        return self.transcriptions.get(file_key, self.default_transcription)


class MockFileSystemProvider(FileSystemProvider):
    """Mock implementation of file system operations for testing."""
    
    def __init__(self):
        self.files = {}  # path -> content mapping
        self.directories = set()  # set of directory paths
        self.file_errors = {}  # path -> exception mapping
    
    def set_file_content(self, path: Path, content: str):
        """Set file content for testing."""
        self.files[str(path)] = content
    
    def set_binary_content(self, path: Path, content: bytes):
        """Set binary file content for testing."""
        self.files[str(path)] = content
    
    def set_file_error(self, path: Path, exception: Exception):
        """Set expected file operation error."""
        self.file_errors[str(path)] = exception
    
    def add_directory(self, path: Path):
        """Add a directory for testing."""
        self.directories.add(str(path))
    
    def read_text(self, path: Path, encoding: str = "utf-8") -> str:
        """Mock read text operation."""
        path_str = str(path)
        
        if path_str in self.file_errors:
            raise self.file_errors[path_str]
        
        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        
        content = self.files[path_str]
        if isinstance(content, bytes):
            return content.decode(encoding)
        return content
    
    def write_text(self, path: Path, content: str, encoding: str = "utf-8") -> None:
        """Mock write text operation."""
        path_str = str(path)
        
        if path_str in self.file_errors:
            raise self.file_errors[path_str]
        
        self.files[path_str] = content
    
    def read_binary(self, path: Path) -> bytes:
        """Mock read binary operation."""
        path_str = str(path)
        
        if path_str in self.file_errors:
            raise self.file_errors[path_str]
        
        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        
        content = self.files[path_str]
        if isinstance(content, str):
            return content.encode()
        return content
    
    def exists(self, path: Path) -> bool:
        """Mock exists check."""
        path_str = str(path)
        return path_str in self.files or path_str in self.directories
    
    def mkdir(self, path: Path, parents: bool = False, exist_ok: bool = False) -> None:
        """Mock directory creation."""
        path_str = str(path)
        
        if path_str in self.file_errors:
            raise self.file_errors[path_str]
        
        if not exist_ok and path_str in self.directories:
            raise FileExistsError(f"Directory exists: {path}")
        
        self.directories.add(path_str)


class MockTimeProvider(TimeProvider):
    """Mock implementation of time operations for testing."""
    
    def __init__(self):
        self.current_time = 1234567890.0  # Fixed time for testing
        self.sleep_calls = []
        self.strftime_calls = []
    
    def set_time(self, time_value: float):
        """Set current time for testing."""
        self.current_time = time_value
    
    def time(self) -> float:
        """Mock current time."""
        return self.current_time
    
    def sleep(self, seconds: float) -> None:
        """Mock sleep operation (just record the call)."""
        self.sleep_calls.append(seconds)
    
    def strftime(self, format_str: str) -> str:
        """Mock strftime operation."""
        self.strftime_calls.append(format_str)
        import time
        return time.strftime(format_str, time.localtime(self.current_time))


class MockEnvironmentProvider(EnvironmentProvider):
    """Mock implementation of environment variable access for testing."""
    
    def __init__(self):
        self.variables = {}  # key -> value mapping
        self.required_errors = {}  # key -> exception mapping
    
    def set_variable(self, key: str, value: str):
        """Set environment variable value for testing."""
        self.variables[key] = value
    
    def set_required_error(self, key: str, exception: Exception):
        """Set expected error for required variable."""
        self.required_errors[key] = exception
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Mock environment variable get."""
        return self.variables.get(key, default)
    
    def get_required(self, key: str) -> str:
        """Mock required environment variable get."""
        if key in self.required_errors:
            raise self.required_errors[key]
        
        if key not in self.variables:
            raise RuntimeError(f"{key} is not set")
        
        return self.variables[key]
