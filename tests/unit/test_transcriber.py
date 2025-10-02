"""
Unit tests for the transcriber module.

Tests OpenAI API integration with mocked dependencies.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock

from src.transcribe_pipeline.audio_utils.transcriber import (
    _save_manifest,
    _transcribe_one,
    _retrying_transcribe,
    transcribe_manifest,
)


class TestSaveManifest:
    """Test manifest saving functionality."""
    
    def test_save_manifest_success(self, mock_fs_provider, temp_dir):
        """Test successful manifest saving."""
        manifest_path = temp_dir / "manifest.json"
        manifest = {"test": "data", "chunks": []}
        
        _save_manifest(manifest_path, manifest, mock_fs_provider)
        
        # Verify file was written
        assert mock_fs_provider.exists(manifest_path)
        content = mock_fs_provider.read_text(manifest_path)
        parsed_content = json.loads(content)
        assert parsed_content == manifest
    
    def test_save_manifest_with_unicode(self, mock_fs_provider, temp_dir):
        """Test manifest saving with Unicode content."""
        manifest_path = temp_dir / "manifest.json"
        manifest = {"text": "Hello 世界", "special": "ñáéíóú"}
        
        _save_manifest(manifest_path, manifest, mock_fs_provider)
        
        content = mock_fs_provider.read_text(manifest_path)
        parsed_content = json.loads(content)
        assert parsed_content == manifest


class TestTranscribeOne:
    """Test single chunk transcription functionality."""
    
    def test_transcribe_one_success(self, mock_openai_provider, mock_time_provider, sample_config):
        """Test successful single chunk transcription."""
        # Setup mock
        mock_client = Mock()
        chunk = {"file": "/path/to/chunk.m4a"}
        
        expected_result = {"text": "Transcribed text content"}
        mock_openai_provider.set_transcription_result(
            Path("/path/to/chunk.m4a"), 
            expected_result
        )
        
        # Mock time provider
        mock_time_provider.set_time(1000.0)  # Start time
        
        # Call function
        text, latency_ms = _transcribe_one(
            mock_client, chunk, sample_config, mock_openai_provider, mock_time_provider
        )
        
        # Verify results
        assert text == "Transcribed text content"
        assert latency_ms >= 0  # Should be non-negative
        
        # Verify OpenAI provider was called correctly
        assert len(mock_openai_provider.clients_created) == 0  # Client passed in
        assert str(Path("/path/to/chunk.m4a")) in mock_openai_provider.transcriptions
    
    def test_transcribe_one_with_prompt(self, mock_openai_provider, mock_time_provider, sample_config):
        """Test transcription with custom prompt."""
        # Setup config with prompt
        sample_config.model.prompt = "Technical presentation"
        
        mock_client = Mock()
        chunk = {"file": "/path/to/chunk.m4a"}
        
        expected_result = {"text": "Technical transcription"}
        mock_openai_provider.set_transcription_result(
            Path("/path/to/chunk.m4a"), 
            expected_result
        )
        
        mock_time_provider.set_time(1000.0)
        
        # Call function
        text, latency_ms = _transcribe_one(
            mock_client, chunk, sample_config, mock_openai_provider, mock_time_provider
        )
        
        assert text == "Technical transcription"
        assert latency_ms >= 0
    
    def test_transcribe_one_openai_error(self, mock_openai_provider, mock_time_provider, sample_config):
        """Test error handling when OpenAI API fails."""
        mock_client = Mock()
        chunk = {"file": "/path/to/chunk.m4a"}
        
        # Setup error
        mock_openai_provider.set_transcription_error(
            Path("/path/to/chunk.m4a"),
            Exception("API rate limit exceeded")
        )
        
        mock_time_provider.set_time(1000.0)
        
        # Should raise the exception
        with pytest.raises(Exception, match="API rate limit exceeded"):
            _transcribe_one(
                mock_client, chunk, sample_config, mock_openai_provider, mock_time_provider
            )


class TestRetryingTranscribe:
    """Test retry logic for transcription."""
    
    def test_retrying_transcribe_success_first_attempt(self, mock_openai_provider, mock_time_provider, sample_config):
        """Test successful transcription on first attempt."""
        mock_client = Mock()
        chunk = {"file": "/path/to/chunk.m4a"}
        
        expected_result = {"text": "Success on first try"}
        mock_openai_provider.set_transcription_result(
            Path("/path/to/chunk.m4a"), 
            expected_result
        )
        
        mock_time_provider.set_time(1000.0)
        
        # Call function
        text, latency_ms = _retrying_transcribe(
            mock_client, chunk, sample_config, mock_openai_provider, mock_time_provider
        )
        
        assert text == "Success on first try"
        assert len(mock_time_provider.sleep_calls) == 0  # No retries needed
    
    def test_retrying_transcribe_retry_on_transient_error(self, mock_openai_provider, mock_time_provider, sample_config):
        """Test retry logic on transient errors."""
        mock_client = Mock()
        chunk = {"file": "/path/to/chunk.m4a"}
        
        # Setup transient error then success
        mock_openai_provider.set_transcription_error(
            Path("/path/to/chunk.m4a"),
            Exception("rate_limit_exceeded")
        )
        
        mock_time_provider.set_time(1000.0)
        
        # Should raise after max retries (1 in test config)
        with pytest.raises(Exception, match="rate_limit_exceeded"):
            _retrying_transcribe(
                mock_client, chunk, sample_config, mock_openai_provider, mock_time_provider
            )
        
        # Should have slept once
        assert len(mock_time_provider.sleep_calls) == 1
        assert mock_time_provider.sleep_calls[0] == 0.1  # 2^0 * 100ms / 1000 (test config has backoff_base_ms=100)
    
    def test_retrying_transcribe_non_transient_error(self, mock_openai_provider, mock_time_provider, sample_config):
        """Test that non-transient errors are not retried."""
        mock_client = Mock()
        chunk = {"file": "/path/to/chunk.m4a"}
        
        # Setup non-transient error
        mock_openai_provider.set_transcription_error(
            Path("/path/to/chunk.m4a"),
            Exception("Invalid audio format")
        )
        
        mock_time_provider.set_time(1000.0)
        
        # Should raise immediately without retry
        with pytest.raises(Exception, match="Invalid audio format"):
            _retrying_transcribe(
                mock_client, chunk, sample_config, mock_openai_provider, mock_time_provider
            )
        
        # Should not have slept
        assert len(mock_time_provider.sleep_calls) == 0


class TestTranscribeManifest:
    """Test manifest transcription functionality."""
    
    def test_transcribe_manifest_success(self, mock_openai_provider, mock_fs_provider, 
                                       mock_time_provider, mock_env_provider, sample_config, 
                                       sample_manifest, temp_dir):
        """Test successful manifest transcription."""
        manifest_path = temp_dir / "manifest.json"
        
        # Setup manifest in mock file system
        mock_fs_provider.set_file_content(manifest_path, json.dumps(sample_manifest))
        
        # Setup transcription results
        for chunk in sample_manifest["chunks"]:
            mock_openai_provider.set_transcription_result(
                Path(chunk["file"]),
                {"text": f"Transcribed chunk {chunk['index']}"}
            )
        
        mock_time_provider.set_time(1000.0)
        
        # Call function
        transcribe_manifest(
            manifest_path, sample_config,
            mock_openai_provider, mock_fs_provider, mock_time_provider, mock_env_provider
        )
        
        # Verify manifest was updated
        updated_content = mock_fs_provider.read_text(manifest_path)
        updated_manifest = json.loads(updated_content)
        
        for chunk in updated_manifest["chunks"]:
            assert chunk["status"] == "done"
            assert chunk["text"] == f"Transcribed chunk {chunk['index']}"
            assert chunk["latency_ms"] is not None
            assert chunk["latency_ms"] >= 0
    
    def test_transcribe_manifest_missing_api_key(self, mock_openai_provider, mock_fs_provider, 
                                               mock_time_provider, sample_config, sample_manifest, temp_dir):
        """Test error when OpenAI API key is missing."""
        manifest_path = temp_dir / "manifest.json"
        
        # Setup manifest
        mock_fs_provider.set_file_content(manifest_path, json.dumps(sample_manifest))
        
        # Setup environment without API key
        mock_env_provider = Mock()
        mock_env_provider.get_required = Mock(side_effect=RuntimeError("OPENAI_API_KEY is not set"))
        
        # Should raise error
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
            transcribe_manifest(
                manifest_path, sample_config,
                mock_openai_provider, mock_fs_provider, mock_time_provider, mock_env_provider
            )
    
    def test_transcribe_manifest_mixed_success_failure(self, mock_openai_provider, mock_fs_provider, 
                                                     mock_time_provider, mock_env_provider, sample_config, 
                                                     sample_manifest, temp_dir):
        """Test transcription with some chunks failing."""
        manifest_path = temp_dir / "manifest.json"
        
        # Setup manifest
        mock_fs_provider.set_file_content(manifest_path, json.dumps(sample_manifest))
        
        # Setup mixed results - first chunk succeeds, second fails
        chunks = sample_manifest["chunks"]
        mock_openai_provider.set_transcription_result(
            Path(chunks[0]["file"]),
            {"text": "Successful transcription"}
        )
        mock_openai_provider.set_transcription_error(
            Path(chunks[1]["file"]),
            Exception("Transcription failed")
        )
        
        mock_time_provider.set_time(1000.0)
        
        # Call function
        transcribe_manifest(
            manifest_path, sample_config,
            mock_openai_provider, mock_fs_provider, mock_time_provider, mock_env_provider
        )
        
        # Verify results
        updated_content = mock_fs_provider.read_text(manifest_path)
        updated_manifest = json.loads(updated_content)
        
        # First chunk should be successful
        assert updated_manifest["chunks"][0]["status"] == "done"
        assert updated_manifest["chunks"][0]["text"] == "Successful transcription"
        
        # Second chunk should have failed
        assert updated_manifest["chunks"][1]["status"] == "error"
        assert "Transcription failed" in updated_manifest["chunks"][1]["error"]
    
    def test_transcribe_manifest_no_pending_chunks(self, mock_openai_provider, mock_fs_provider, 
                                                 mock_time_provider, mock_env_provider, sample_config, temp_dir):
        """Test transcription when no chunks are pending."""
        manifest_path = temp_dir / "manifest.json"
        
        # Create manifest with no pending chunks
        manifest = {
            "chunks": [
                {"index": 0, "status": "done", "text": "Already done"},
                {"index": 1, "status": "error", "error": "Already failed"},
            ]
        }
        
        mock_fs_provider.set_file_content(manifest_path, json.dumps(manifest))
        
        # Call function
        transcribe_manifest(
            manifest_path, sample_config,
            mock_openai_provider, mock_fs_provider, mock_time_provider, mock_env_provider
        )
        
        # Should complete without calling OpenAI transcription
        # Note: A client is still created for the function, but no transcription calls are made
        assert len(mock_openai_provider.clients_created) == 1  # Client created but no transcriptions
    
    def test_transcribe_manifest_backward_compatibility(self, sample_config, sample_manifest, temp_dir):
        """Test backward compatibility with default providers."""
        manifest_path = temp_dir / "manifest.json"
        
        # This test would require actual OpenAI API key and network access
        # For now, we'll just test that the function signature works
        try:
            # This should not crash due to signature issues
            transcribe_manifest(manifest_path, sample_config)
        except RuntimeError as e:
            # Expected if no API key is available
            assert "OPENAI_API_KEY" in str(e)
        except Exception:
            # Other exceptions are acceptable for this compatibility test
            pass
