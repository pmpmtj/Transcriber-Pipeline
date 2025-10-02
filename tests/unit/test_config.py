"""
Unit tests for the pipeline configuration module.

Tests configuration validation, CLI overrides, and edge cases.
"""

import pytest
from pathlib import Path

from src.transcribe_pipeline.config.pipeline_config import (
    PipelineConfig,
    ModelConfig,
    ChunkingConfig,
    ReencodeConfig,
    SilenceConfig,
    OutputConfig,
    PathConfig,
    create_default_config,
    create_config_from_dict,
)


class TestModelConfig:
    """Test ModelConfig validation and behavior."""
    
    def test_default_model_config(self):
        """Test default model configuration values."""
        config = ModelConfig()
        assert config.model == "gpt-4o-transcribe"
        assert config.response_format == "json"
        assert config.prompt == ""
        assert config.parallel_requests == 3
        assert config.max_retries == 3
        assert config.backoff_base_ms == 800
    
    def test_valid_model_values(self):
        """Test valid model configuration values."""
        config = ModelConfig(
            model="gpt-4o-mini-transcribe",
            response_format="text",
            parallel_requests=5,
            max_retries=2,
            backoff_base_ms=1000
        )
        config.validate()  # Should not raise
    
    def test_invalid_model(self):
        """Test invalid model validation."""
        config = ModelConfig(model="invalid-model")
        with pytest.raises(ValueError, match="Invalid model"):
            config.validate()
    
    def test_invalid_response_format(self):
        """Test invalid response format validation."""
        config = ModelConfig(response_format="invalid")
        with pytest.raises(ValueError, match="Invalid response_format"):
            config.validate()
    
    def test_invalid_parallel_requests(self):
        """Test invalid parallel requests validation."""
        config = ModelConfig(parallel_requests=0)
        with pytest.raises(ValueError, match="parallel_requests must be between 1 and 10"):
            config.validate()
        
        config = ModelConfig(parallel_requests=11)
        with pytest.raises(ValueError, match="parallel_requests must be between 1 and 10"):
            config.validate()
    
    def test_invalid_max_retries(self):
        """Test invalid max retries validation."""
        config = ModelConfig(max_retries=-1)
        with pytest.raises(ValueError, match="max_retries must be between 0 and 10"):
            config.validate()
    
    def test_invalid_backoff_base_ms(self):
        """Test invalid backoff base ms validation."""
        config = ModelConfig(backoff_base_ms=50)
        with pytest.raises(ValueError, match="backoff_base_ms must be between 100 and 5000"):
            config.validate()


class TestChunkingConfig:
    """Test ChunkingConfig validation and behavior."""
    
    def test_default_chunking_config(self):
        """Test default chunking configuration values."""
        config = ChunkingConfig()
        assert config.max_file_mb == 25
        assert config.target_chunk_mb == 16
        assert config.max_chunk_secs == 900
        assert config.overlap_secs == 3.0
    
    def test_valid_chunking_values(self):
        """Test valid chunking configuration values."""
        config = ChunkingConfig(
            max_file_mb=50,
            target_chunk_mb=20,
            max_chunk_secs=600,
            overlap_secs=2.0
        )
        config.validate()  # Should not raise
    
    def test_invalid_max_file_mb(self):
        """Test invalid max file mb validation."""
        config = ChunkingConfig(max_file_mb=0)
        with pytest.raises(ValueError, match="max_file_mb must be between 1 and 100"):
            config.validate()
    
    def test_invalid_target_chunk_mb(self):
        """Test invalid target chunk mb validation."""
        config = ChunkingConfig(target_chunk_mb=30, max_file_mb=25)
        with pytest.raises(ValueError, match="target_chunk_mb must be between 1 and 25"):
            config.validate()
    
    def test_invalid_max_chunk_secs(self):
        """Test invalid max chunk secs validation."""
        config = ChunkingConfig(max_chunk_secs=30)
        with pytest.raises(ValueError, match="max_chunk_secs must be between 60 and 3600"):
            config.validate()
    
    def test_invalid_overlap_secs(self):
        """Test invalid overlap secs validation."""
        config = ChunkingConfig(overlap_secs=35.0)
        with pytest.raises(ValueError, match="overlap_secs must be between 0 and 30"):
            config.validate()


class TestReencodeConfig:
    """Test ReencodeConfig validation and behavior."""
    
    def test_default_reencode_config(self):
        """Test default re-encoding configuration values."""
        config = ReencodeConfig()
        assert config.enabled is True
        assert config.codec == "aac"
        assert config.bitrate_kbps == 64
        assert config.channels == 1
        assert config.sample_rate == 16000
    
    def test_valid_reencode_values(self):
        """Test valid re-encoding configuration values."""
        config = ReencodeConfig(
            codec="mp3",
            bitrate_kbps=128,
            channels=2,
            sample_rate=44100
        )
        config.validate()  # Should not raise
    
    def test_invalid_codec(self):
        """Test invalid codec validation."""
        config = ReencodeConfig(codec="invalid")
        with pytest.raises(ValueError, match="Invalid codec"):
            config.validate()
    
    def test_invalid_bitrate(self):
        """Test invalid bitrate validation."""
        config = ReencodeConfig(bitrate_kbps=20)
        with pytest.raises(ValueError, match="bitrate_kbps must be between 32 and 320"):
            config.validate()
    
    def test_invalid_channels(self):
        """Test invalid channels validation."""
        config = ReencodeConfig(channels=3)
        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            config.validate()
    
    def test_invalid_sample_rate(self):
        """Test invalid sample rate validation."""
        config = ReencodeConfig(sample_rate=11025)
        with pytest.raises(ValueError, match="Invalid sample_rate"):
            config.validate()


class TestPipelineConfig:
    """Test PipelineConfig integration and behavior."""
    
    def test_default_pipeline_config(self):
        """Test default pipeline configuration."""
        config = create_default_config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.reencode, ReencodeConfig)
        assert isinstance(config.silence, SilenceConfig)
        assert isinstance(config.outputs, OutputConfig)
        assert isinstance(config.paths, PathConfig)
        assert config.segmenter == "fixed"
    
    def test_config_validation(self):
        """Test complete configuration validation."""
        config = create_default_config()
        config.validate()  # Should not raise
    
    def test_invalid_segmenter(self):
        """Test invalid segmenter validation."""
        config = create_default_config()
        config.segmenter = "invalid"
        with pytest.raises(ValueError, match="Invalid segmenter"):
            config.validate()
    
    def test_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = create_default_config()
        config_dict = config.to_dict()
        
        assert "model" in config_dict
        assert "max_file_mb" in config_dict  # chunking fields are flattened
        assert "reencode" in config_dict
        assert "outputs" in config_dict
        assert config_dict["model"] == "gpt-4o-transcribe"
        assert config_dict["segmenter"] == "fixed"
    
    def test_apply_cli_overrides(self):
        """Test CLI argument override application."""
        config = create_default_config()
        
        # Create mock args object
        class MockArgs:
            model = "gpt-4o-mini-transcribe"
            segmenter = "silence"
            prompt = "Test prompt"
            work_dir = "/test/workdir"
        
        args = MockArgs()
        config.apply_cli_overrides(args)
        
        assert config.model.model == "gpt-4o-mini-transcribe"
        assert config.segmenter == "silence"
        assert config.model.prompt == "Test prompt"
        assert config.paths.work_dir == "/test/workdir"
    
    def test_get_input_audio_path_with_cli(self, temp_dir):
        """Test getting input audio path from CLI argument."""
        config = create_default_config()
        test_file = temp_dir / "test.mp3"
        test_file.write_bytes(b"dummy content")
        
        input_path = config.get_input_audio_path(str(test_file))
        
        assert input_path == test_file.resolve()
    
    def test_get_input_audio_path_from_config(self, temp_dir):
        """Test getting input audio path from config."""
        config = create_default_config()
        test_file = temp_dir / "config.mp3"
        test_file.write_bytes(b"dummy content")
        config.paths.input_audio = str(test_file)
        
        input_path = config.get_input_audio_path()
        
        assert input_path == test_file.resolve()
    
    def test_get_input_audio_path_no_input(self):
        """Test error when no input audio is specified."""
        config = create_default_config()
        with pytest.raises(ValueError, match="No input audio specified"):
            config.get_input_audio_path()
    
    def test_get_work_directory(self):
        """Test getting work directory path."""
        config = create_default_config()
        work_dir = config.get_work_directory("/custom/workdir")
        
        assert work_dir == Path("/custom/workdir").expanduser().resolve()


class TestCreateConfigFromDict:
    """Test configuration creation from dictionary."""
    
    def test_create_from_dict_basic(self):
        """Test basic configuration creation from dictionary."""
        config_dict = {
            "model": "gpt-4o-mini-transcribe",
            "segmenter": "silence",
            "max_file_mb": 50,
            "target_chunk_mb": 20,
        }
        
        config = create_config_from_dict(config_dict)
        
        assert config.model.model == "gpt-4o-mini-transcribe"
        assert config.segmenter == "silence"
        assert config.chunking.max_file_mb == 50
        assert config.chunking.target_chunk_mb == 20
    
    def test_create_from_dict_nested(self):
        """Test configuration creation with nested dictionaries."""
        config_dict = {
            "model": "gpt-4o-transcribe",
            "reencode": {
                "enabled": False,
                "codec": "mp3",
                "bitrate_kbps": 128,
                "channels": 2,
                "sample_rate": 44100,
            },
            "outputs": {
                "write_txt": True,
                "write_json": False,
                "write_srt": True,
                "write_vtt": False,
            },
        }
        
        config = create_config_from_dict(config_dict)
        
        assert config.model.model == "gpt-4o-transcribe"
        assert config.reencode.enabled is False
        assert config.reencode.codec == "mp3"
        assert config.outputs.write_json is False
        assert config.outputs.write_srt is True
    
    def test_create_from_dict_validation(self):
        """Test that invalid dictionary values are caught during validation."""
        config_dict = {
            "model": "invalid-model",
            "max_file_mb": 0,  # Invalid
        }
        
        with pytest.raises(ValueError):
            create_config_from_dict(config_dict)
