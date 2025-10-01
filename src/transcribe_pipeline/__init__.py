"""
Transcribe Pipeline

A robust audio transcription pipeline that chunks, transcribes, and stitches audio files
using OpenAI's Whisper API with configurable settings and professional logging.
"""

__version__ = "1.0.0"
__author__ = "Pedro Julio"
__email__ = "pmpmtj@hotmail.com"

from .config.pipeline_config import PipelineConfig, create_default_config, create_config_from_dict

__all__ = [
    "PipelineConfig",
    "create_default_config", 
    "create_config_from_dict",
]
