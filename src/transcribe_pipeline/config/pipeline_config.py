"""
Pipeline Configuration Module

This module provides a comprehensive class-based configuration system for the transcription pipeline.
It replaces the previous YAML-based configuration with type-safe, validated Python classes.

The configuration supports CLI overrides and integrates with the logging system.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from ..utils.path_utils import get_script_directory
from ..logging_utils.logging_config import get_logger


# ============================================================================
# NESTED CONFIGURATION CLASSES
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for OpenAI model and API settings."""
    model: str = "gpt-4o-transcribe"
    response_format: str = "json"
    prompt: str = ""
    parallel_requests: int = 3
    max_retries: int = 3
    backoff_base_ms: int = 800

    def validate(self) -> None:
        """Validate model configuration settings."""
        if self.model not in ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]:
            raise ValueError(f"Invalid model: {self.model}. Must be 'gpt-4o-transcribe' or 'gpt-4o-mini-transcribe'")
        
        if self.response_format not in ["json", "text"]:
            raise ValueError(f"Invalid response_format: {self.response_format}. Must be 'json' or 'text'")
        
        if self.parallel_requests < 1 or self.parallel_requests > 10:
            raise ValueError(f"parallel_requests must be between 1 and 10, got {self.parallel_requests}")
        
        if self.max_retries < 0 or self.max_retries > 10:
            raise ValueError(f"max_retries must be between 0 and 10, got {self.max_retries}")
        
        if self.backoff_base_ms < 100 or self.backoff_base_ms > 5000:
            raise ValueError(f"backoff_base_ms must be between 100 and 5000, got {self.backoff_base_ms}")


@dataclass
class ChunkingConfig:
    """Configuration for audio chunking policy."""
    max_file_mb: int = 25
    target_chunk_mb: int = 16
    max_chunk_secs: int = 900  # 15 minutes
    overlap_secs: float = 3.0

    def validate(self) -> None:
        """Validate chunking configuration settings."""
        if self.max_file_mb < 1 or self.max_file_mb > 100:
            raise ValueError(f"max_file_mb must be between 1 and 100, got {self.max_file_mb}")
        
        if self.target_chunk_mb < 1 or self.target_chunk_mb > self.max_file_mb:
            raise ValueError(f"target_chunk_mb must be between 1 and {self.max_file_mb}, got {self.target_chunk_mb}")
        
        if self.max_chunk_secs < 60 or self.max_chunk_secs > 3600:
            raise ValueError(f"max_chunk_secs must be between 60 and 3600, got {self.max_chunk_secs}")
        
        if self.overlap_secs < 0 or self.overlap_secs > 30:
            raise ValueError(f"overlap_secs must be between 0 and 30, got {self.overlap_secs}")


@dataclass
class ReencodeConfig:
    """Configuration for audio re-encoding settings."""
    enabled: bool = True
    codec: str = "aac"
    bitrate_kbps: int = 64
    channels: int = 1
    sample_rate: int = 16000

    def validate(self) -> None:
        """Validate re-encoding configuration settings."""
        if self.codec not in ["aac", "libfdk_aac", "mp3", "wav"]:
            raise ValueError(f"Invalid codec: {self.codec}. Must be one of: aac, libfdk_aac, mp3, wav")
        
        if self.bitrate_kbps < 32 or self.bitrate_kbps > 320:
            raise ValueError(f"bitrate_kbps must be between 32 and 320, got {self.bitrate_kbps}")
        
        if self.channels < 1 or self.channels > 2:
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")
        
        if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}. Must be one of: 8000, 16000, 22050, 44100, 48000")


@dataclass
class SilenceConfig:
    """Configuration for silence-based segmentation."""
    min_silence_db: int = -35
    min_silence_dur: float = 0.6

    def validate(self) -> None:
        """Validate silence detection configuration settings."""
        if self.min_silence_db > -10 or self.min_silence_db < -60:
            raise ValueError(f"min_silence_db must be between -60 and -10, got {self.min_silence_db}")
        
        if self.min_silence_dur < 0.1 or self.min_silence_dur > 5.0:
            raise ValueError(f"min_silence_dur must be between 0.1 and 5.0, got {self.min_silence_dur}")


@dataclass
class OutputConfig:
    """Configuration for output file generation."""
    write_txt: bool = True
    write_json: bool = True
    write_srt: bool = True
    write_vtt: bool = False

    def validate(self) -> None:
        """Validate output configuration settings."""
        # All boolean values are valid by default
        pass


@dataclass
class PathConfig:
    """Configuration for file and directory paths."""
    input_audio: str = ""
    work_dir: str = "outputs"

    def validate(self) -> None:
        """Validate path configuration settings."""
        if self.work_dir and not self.work_dir.strip():
            raise ValueError("work_dir cannot be empty")
        
        # input_audio can be empty (set via CLI)


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

@dataclass
class PipelineConfig:
    """
    Main configuration class for the transcription pipeline.
    
    This class contains all configuration settings organized into logical groups.
    It supports CLI overrides and integrates with the logging system.
    """
    
    # Nested configuration objects
    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    reencode: ReencodeConfig = field(default_factory=ReencodeConfig)
    silence: SilenceConfig = field(default_factory=SilenceConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Additional settings
    segmenter: str = "fixed"  # fixed | silence
    
    def __post_init__(self):
        """Initialize logger and validate configuration after instantiation."""
        self.logger = get_logger("pipeline_config")
        self.validate()
        self.logger.debug("PipelineConfig initialized successfully")

    def validate(self) -> None:
        """Validate all configuration settings."""
        try:
            self.model.validate()
            self.chunking.validate()
            self.reencode.validate()
            self.silence.validate()
            self.outputs.validate()
            self.paths.validate()
            
            if self.segmenter not in ["fixed", "silence"]:
                raise ValueError(f"Invalid segmenter: {self.segmenter}. Must be 'fixed' or 'silence'")
                
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format for backward compatibility."""
        return {
            "model": self.model.model,
            "response_format": self.model.response_format,
            "prompt": self.model.prompt,
            "parallel_requests": self.model.parallel_requests,
            "max_retries": self.model.max_retries,
            "backoff_base_ms": self.model.backoff_base_ms,
            "max_file_mb": self.chunking.max_file_mb,
            "target_chunk_mb": self.chunking.target_chunk_mb,
            "max_chunk_secs": self.chunking.max_chunk_secs,
            "overlap_secs": self.chunking.overlap_secs,
            "reencode": {
                "enabled": self.reencode.enabled,
                "codec": self.reencode.codec,
                "bitrate_kbps": self.reencode.bitrate_kbps,
                "channels": self.reencode.channels,
                "sample_rate": self.reencode.sample_rate,
            },
            "segmenter": self.segmenter,
            "silence": {
                "min_silence_db": self.silence.min_silence_db,
                "min_silence_dur": self.silence.min_silence_dur,
            },
            "outputs": {
                "write_txt": self.outputs.write_txt,
                "write_json": self.outputs.write_json,
                "write_srt": self.outputs.write_srt,
                "write_vtt": self.outputs.write_vtt,
            },
            "input_audio": self.paths.input_audio,
            "work_dir": self.paths.work_dir,
        }

    def apply_cli_overrides(self, args) -> None:
        """Apply CLI argument overrides to configuration."""
        if hasattr(args, 'model') and args.model:
            self.model.model = args.model
            self.logger.debug(f"CLI override: model = {args.model}")
        
        if hasattr(args, 'segmenter') and args.segmenter:
            self.segmenter = args.segmenter
            self.logger.debug(f"CLI override: segmenter = {args.segmenter}")
        
        if hasattr(args, 'prompt') and args.prompt is not None:
            self.model.prompt = args.prompt
            self.logger.debug(f"CLI override: prompt = {args.prompt}")
        
        if hasattr(args, 'work_dir') and args.work_dir:
            self.paths.work_dir = args.work_dir
            self.logger.debug(f"CLI override: work_dir = {args.work_dir}")
        
        # Re-validate after applying overrides
        self.validate()
        self.logger.info("CLI overrides applied and validated successfully")

    def get_input_audio_path(self, cli_input: Optional[str] = None) -> Path:
        """Get the input audio path, prioritizing CLI argument over config."""
        input_audio = cli_input or self.paths.input_audio
        if not input_audio:
            raise ValueError("No input audio specified. Provide --input or set input_audio in config.")
        
        input_path = Path(input_audio).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
        return input_path

    def get_work_directory(self, cli_work_dir: Optional[str] = None) -> Path:
        """Get the work directory path, prioritizing CLI argument over config."""
        work_dir = cli_work_dir or self.paths.work_dir
        return Path(work_dir).expanduser().resolve()

    def save_effective_config(self, output_dir: Path) -> None:
        """Save the effective configuration (including CLI overrides) to a file."""
        effective_config = self.to_dict()
        config_file = output_dir / "effective_config.json"
        
        import json
        config_file.write_text(
            json.dumps(effective_config, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        self.logger.debug(f"Effective configuration saved to: {config_file}")

    def log_configuration(self) -> None:
        """Log the current configuration for debugging purposes."""
        self.logger.debug("Current pipeline configuration:")
        self.logger.debug(f"  Model: {self.model.model} ({self.model.response_format})")
        self.logger.debug(f"  Chunking: {self.chunking.target_chunk_mb}MB target, {self.chunking.max_chunk_secs}s max")
        self.logger.debug(f"  Segmenter: {self.segmenter}")
        self.logger.debug(f"  Re-encoding: {'enabled' if self.reencode.enabled else 'disabled'}")
        self.logger.debug(f"  Outputs: txt={self.outputs.write_txt}, json={self.outputs.write_json}, srt={self.outputs.write_srt}, vtt={self.outputs.write_vtt}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_default_config() -> PipelineConfig:
    """Create a new PipelineConfig instance with default values."""
    return PipelineConfig()


def create_config_from_dict(config_dict: Dict[str, Any]) -> PipelineConfig:
    """Create a PipelineConfig instance from a dictionary."""
    config = PipelineConfig()
    
    # Update model settings
    if "model" in config_dict:
        config.model.model = config_dict["model"]
    if "response_format" in config_dict:
        config.model.response_format = config_dict["response_format"]
    if "prompt" in config_dict:
        config.model.prompt = config_dict["prompt"]
    if "parallel_requests" in config_dict:
        config.model.parallel_requests = config_dict["parallel_requests"]
    if "max_retries" in config_dict:
        config.model.max_retries = config_dict["max_retries"]
    if "backoff_base_ms" in config_dict:
        config.model.backoff_base_ms = config_dict["backoff_base_ms"]
    
    # Update chunking settings
    if "max_file_mb" in config_dict:
        config.chunking.max_file_mb = config_dict["max_file_mb"]
    if "target_chunk_mb" in config_dict:
        config.chunking.target_chunk_mb = config_dict["target_chunk_mb"]
    if "max_chunk_secs" in config_dict:
        config.chunking.max_chunk_secs = config_dict["max_chunk_secs"]
    if "overlap_secs" in config_dict:
        config.chunking.overlap_secs = config_dict["overlap_secs"]
    
    # Update re-encoding settings
    if "reencode" in config_dict:
        reencode_dict = config_dict["reencode"]
        if "enabled" in reencode_dict:
            config.reencode.enabled = reencode_dict["enabled"]
        if "codec" in reencode_dict:
            config.reencode.codec = reencode_dict["codec"]
        if "bitrate_kbps" in reencode_dict:
            config.reencode.bitrate_kbps = reencode_dict["bitrate_kbps"]
        if "channels" in reencode_dict:
            config.reencode.channels = reencode_dict["channels"]
        if "sample_rate" in reencode_dict:
            config.reencode.sample_rate = reencode_dict["sample_rate"]
    
    # Update segmenter
    if "segmenter" in config_dict:
        config.segmenter = config_dict["segmenter"]
    
    # Update silence settings
    if "silence" in config_dict:
        silence_dict = config_dict["silence"]
        if "min_silence_db" in silence_dict:
            config.silence.min_silence_db = silence_dict["min_silence_db"]
        if "min_silence_dur" in silence_dict:
            config.silence.min_silence_dur = silence_dict["min_silence_dur"]
    
    # Update output settings
    if "outputs" in config_dict:
        outputs_dict = config_dict["outputs"]
        if "write_txt" in outputs_dict:
            config.outputs.write_txt = outputs_dict["write_txt"]
        if "write_json" in outputs_dict:
            config.outputs.write_json = outputs_dict["write_json"]
        if "write_srt" in outputs_dict:
            config.outputs.write_srt = outputs_dict["write_srt"]
        if "write_vtt" in outputs_dict:
            config.outputs.write_vtt = outputs_dict["write_vtt"]
    
    # Update path settings
    if "input_audio" in config_dict:
        config.paths.input_audio = config_dict["input_audio"]
    if "work_dir" in config_dict:
        config.paths.work_dir = config_dict["work_dir"]
    
    config.validate()
    return config
