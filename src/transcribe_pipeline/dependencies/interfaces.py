"""
Dependency Injection Interfaces

This module defines abstract base classes for all external dependencies
used by the transcription pipeline. These interfaces allow for easy
mocking and testing while maintaining clear contracts.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess


class SubprocessProvider(ABC):
    """Interface for subprocess operations (FFmpeg calls)."""
    
    @abstractmethod
    def run(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        """
        Run a subprocess command.
        
        Args:
            cmd: Command and arguments as list
            **kwargs: Additional subprocess.run() arguments
            
        Returns:
            CompletedProcess result
            
        Raises:
            RuntimeError: If command fails
        """
        pass


class OpenAIClientProvider(ABC):
    """Interface for OpenAI API operations."""
    
    @abstractmethod
    def create_client(self, api_key: str) -> Any:
        """
        Create an OpenAI client instance.
        
        Args:
            api_key: OpenAI API key
            
        Returns:
            OpenAI client instance
        """
        pass
    
    @abstractmethod
    def transcribe_audio(self, client: Any, audio_file: Path, model: str, 
                        response_format: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI API.
        
        Args:
            client: OpenAI client instance
            audio_file: Path to audio file
            model: Model to use (e.g., 'gpt-4o-transcribe')
            response_format: Response format ('json' or 'text')
            prompt: Optional transcription prompt
            
        Returns:
            Transcription result dictionary
            
        Raises:
            Exception: If transcription fails
        """
        pass


class FileSystemProvider(ABC):
    """Interface for file system operations."""
    
    @abstractmethod
    def read_text(self, path: Path, encoding: str = "utf-8") -> str:
        """
        Read text from a file.
        
        Args:
            path: Path to file
            encoding: File encoding
            
        Returns:
            File contents as string
        """
        pass
    
    @abstractmethod
    def write_text(self, path: Path, content: str, encoding: str = "utf-8") -> None:
        """
        Write text to a file.
        
        Args:
            path: Path to file
            content: Content to write
            encoding: File encoding
        """
        pass
    
    @abstractmethod
    def read_binary(self, path: Path) -> bytes:
        """
        Read binary data from a file.
        
        Args:
            path: Path to file
            
        Returns:
            File contents as bytes
        """
        pass
    
    @abstractmethod
    def exists(self, path: Path) -> bool:
        """
        Check if a path exists.
        
        Args:
            path: Path to check
            
        Returns:
            True if path exists, False otherwise
        """
        pass
    
    @abstractmethod
    def mkdir(self, path: Path, parents: bool = False, exist_ok: bool = False) -> None:
        """
        Create a directory.
        
        Args:
            path: Directory path
            parents: Create parent directories
            exist_ok: Don't raise error if directory exists
        """
        pass


class TimeProvider(ABC):
    """Interface for time-related operations."""
    
    @abstractmethod
    def time(self) -> float:
        """
        Get current time as Unix timestamp.
        
        Returns:
            Current time as float
        """
        pass
    
    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """
        Sleep for specified seconds.
        
        Args:
            seconds: Seconds to sleep
        """
        pass
    
    @abstractmethod
    def strftime(self, format_str: str) -> str:
        """
        Format current time as string.
        
        Args:
            format_str: Time format string
            
        Returns:
            Formatted time string
        """
        pass


class EnvironmentProvider(ABC):
    """Interface for environment variable access."""
    
    @abstractmethod
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value or default
        """
        pass
    
    @abstractmethod
    def get_required(self, key: str) -> str:
        """
        Get required environment variable value.
        
        Args:
            key: Environment variable name
            
        Returns:
            Environment variable value
            
        Raises:
            RuntimeError: If environment variable is not set
        """
        pass
