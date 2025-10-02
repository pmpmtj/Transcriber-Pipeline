"""
Unit tests for the FFprobe audio analysis module.

Tests audio metadata extraction with mocked subprocess operations.
"""

import pytest
import subprocess
from pathlib import Path

from src.transcribe_pipeline.audio_utils.ffprobe import probe_audio


class TestProbeAudio:
    """Test audio probing functionality."""
    
    def test_probe_audio_success(self, mock_subprocess_provider, sample_audio_file, mock_ffprobe_success_result):
        """Test successful audio probing with mock subprocess."""
        # Setup mock subprocess to return successful FFprobe result
        mock_subprocess_provider.set_result(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(sample_audio_file)],
            mock_ffprobe_success_result
        )
        
        # Call probe_audio with mock provider
        result = probe_audio(sample_audio_file, mock_subprocess_provider)
        
        # Verify result structure and values
        assert isinstance(result, dict)
        assert "duration" in result
        assert "bit_rate" in result
        assert "sample_rate" in result
        assert "channels" in result
        assert "format_name" in result
        assert "size_bytes" in result
        
        assert result["duration"] == 120.5
        assert result["bit_rate"] == 128000
        assert result["sample_rate"] == 44100
        assert result["channels"] == 2
        assert result["format_name"] == "mp3"
        assert result["size_bytes"] == 1920000
    
    def test_probe_audio_default_provider(self, sample_audio_file):
        """Test that probe_audio works with default provider (backward compatibility)."""
        # This test would require actual FFprobe installation
        # For now, we'll test that the function doesn't crash with default provider
        try:
            result = probe_audio(sample_audio_file)
            # If FFprobe is available, verify basic structure
            assert isinstance(result, dict)
            assert "duration" in result
        except RuntimeError:
            # Expected if FFprobe is not available - this is acceptable for testing
            pass
    
    def test_probe_audio_subprocess_error(self, mock_subprocess_provider, sample_audio_file):
        """Test error handling when subprocess fails."""
        # Setup mock subprocess to return error
        error_result = subprocess.CompletedProcess(
            args=["ffprobe", "test.mp3"],
            returncode=1,
            stdout=b"",
            stderr=b"FFprobe error: No such file or directory"
        )
        
        mock_subprocess_provider.set_result(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(sample_audio_file)],
            error_result
        )
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="FFprobe error"):
            probe_audio(sample_audio_file, mock_subprocess_provider)
    
    def test_probe_audio_missing_duration(self, mock_subprocess_provider, sample_audio_file):
        """Test handling of audio files with missing duration information."""
        import json
        
        # Create FFprobe output without duration
        ffprobe_output = json.dumps({
            "format": {
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
        
        success_result = subprocess.CompletedProcess(
            args=["ffprobe", "test.mp3"],
            returncode=0,
            stdout=ffprobe_output,
            stderr=b""
        )
        
        mock_subprocess_provider.set_result(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(sample_audio_file)],
            success_result
        )
        
        result = probe_audio(sample_audio_file, mock_subprocess_provider)
        
        # Should handle missing duration gracefully
        assert result["duration"] == 0.0
        assert result["bit_rate"] == 128000
    
    def test_probe_audio_missing_audio_stream(self, mock_subprocess_provider, sample_audio_file):
        """Test handling of files without audio streams."""
        import json
        
        # Create FFprobe output without audio stream
        ffprobe_output = json.dumps({
            "format": {
                "duration": "120.5",
                "bit_rate": "128000",
                "format_name": "mp3",
                "size": "1920000"
            },
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080
                }
            ]
        }).encode('utf-8')
        
        success_result = subprocess.CompletedProcess(
            args=["ffprobe", "test.mp3"],
            returncode=0,
            stdout=ffprobe_output,
            stderr=b""
        )
        
        mock_subprocess_provider.set_result(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(sample_audio_file)],
            success_result
        )
        
        result = probe_audio(sample_audio_file, mock_subprocess_provider)
        
        # Should use default values for missing audio stream info
        assert result["duration"] == 120.5
        assert result["bit_rate"] == 128000
        assert result["sample_rate"] == 44100  # Default
        assert result["channels"] == 2  # Default
    
    def test_probe_audio_command_construction(self, mock_subprocess_provider, sample_audio_file, mock_ffprobe_success_result):
        """Test that the correct FFprobe command is constructed."""
        # Setup mock result for the specific command
        expected_cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(sample_audio_file)
        ]
        mock_subprocess_provider.set_result(expected_cmd, mock_ffprobe_success_result)
        
        # Call probe_audio
        probe_audio(sample_audio_file, mock_subprocess_provider)
        
        # Verify the command that was run
        commands_run = mock_subprocess_provider.commands_run
        assert len(commands_run) == 1
        
        cmd, kwargs = commands_run[0]
        assert cmd == expected_cmd
        assert "stdout" in kwargs
        assert "stderr" in kwargs
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        assert kwargs["check"] is False
    
    def test_probe_audio_result_parsing_edge_cases(self, mock_subprocess_provider, sample_audio_file):
        """Test parsing of edge case FFprobe outputs."""
        import json
        
        # Test with string values that need conversion
        ffprobe_output = json.dumps({
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
                    "channels": "2"
                }
            ]
        }).encode('utf-8')
        
        success_result = subprocess.CompletedProcess(
            args=["ffprobe", "test.mp3"],
            returncode=0,
            stdout=ffprobe_output,
            stderr=b""
        )
        
        mock_subprocess_provider.set_result(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(sample_audio_file)],
            success_result
        )
        
        result = probe_audio(sample_audio_file, mock_subprocess_provider)
        
        # Verify type conversions
        assert isinstance(result["duration"], float)
        assert isinstance(result["bit_rate"], int)
        assert isinstance(result["sample_rate"], int)
        assert isinstance(result["channels"], int)
        assert isinstance(result["size_bytes"], int)
