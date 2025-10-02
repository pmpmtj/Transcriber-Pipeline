import concurrent.futures
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from tqdm import tqdm

from ..dependencies.interfaces import (
    OpenAIClientProvider,
    FileSystemProvider,
    TimeProvider,
    EnvironmentProvider,
)
from ..dependencies.implementations import (
    RealOpenAIClientProvider,
    RealFileSystemProvider,
    RealTimeProvider,
    RealEnvironmentProvider,
)


TRANSIENT_ERRORS = ("rate_limit_exceeded", "server_error", "temporarily_unavailable")


def _save_manifest(manifest_path: Path, manifest: dict, fs_provider: FileSystemProvider):
    """Save manifest to file using the provided file system provider."""
    content = json.dumps(manifest, indent=2, ensure_ascii=False)
    fs_provider.write_text(manifest_path, content, encoding="utf-8")


def _transcribe_one(
    client: Any, 
    chunk: dict, 
    config: Any,
    openai_provider: OpenAIClientProvider,
    time_provider: TimeProvider
) -> tuple[str, int]:
    """Transcribe a single audio chunk using the provided providers."""
    start = time_provider.time()
    model = config.model.model
    response_format = config.model.response_format
    prompt = config.model.prompt

    audio_file = Path(chunk["file"])
    data = openai_provider.transcribe_audio(
        client=client,
        audio_file=audio_file,
        model=model,
        response_format=response_format,
        prompt=prompt if prompt else None,
    )

    text = data.get("text") if isinstance(data, dict) else None
    latency_ms = int((time_provider.time() - start) * 1000)
    return text, latency_ms


def _retrying_transcribe(
    client: Any,
    chunk: dict,
    config: Any,
    openai_provider: OpenAIClientProvider,
    time_provider: TimeProvider
) -> tuple[str, int]:
    """Transcribe with retry logic using the provided providers."""
    max_retries = config.model.max_retries
    backoff = config.model.backoff_base_ms

    for attempt in range(max_retries + 1):
        try:
            return _transcribe_one(client, chunk, config, openai_provider, time_provider)
        except Exception as e:
            msg = str(e).lower()
            if attempt < max_retries and any(t in msg for t in TRANSIENT_ERRORS):
                delay = (2 ** attempt) * backoff / 1000.0
                time_provider.sleep(delay)
                continue
            raise


def transcribe_manifest(
    manifest_path: Path, 
    config: Any,
    openai_provider: Optional[OpenAIClientProvider] = None,
    fs_provider: Optional[FileSystemProvider] = None,
    time_provider: Optional[TimeProvider] = None,
    env_provider: Optional[EnvironmentProvider] = None
):
    """Transcribe all pending chunks in the manifest using dependency injection."""
    # Use default providers if not provided (for backward compatibility)
    if openai_provider is None:
        openai_provider = RealOpenAIClientProvider()
    if fs_provider is None:
        fs_provider = RealFileSystemProvider()
    if time_provider is None:
        time_provider = RealTimeProvider()
    if env_provider is None:
        env_provider = RealEnvironmentProvider()
    
    # Get API key using environment provider
    api_key = env_provider.get_required("OPENAI_API_KEY")
    
    # Create OpenAI client using provider
    client = openai_provider.create_client(api_key)
    
    # Read manifest using file system provider
    manifest_content = fs_provider.read_text(manifest_path, encoding="utf-8")
    manifest = json.loads(manifest_content)

    pending = [c for c in manifest["chunks"] if c.get("status") == "pending"]
    pbar = tqdm(total=len(pending), desc="Transcribing", unit="chunk")

    def work(c):
        try:
            text, latency = _retrying_transcribe(
                client, c, config, openai_provider, time_provider
            )
            c["text"] = text
            c["latency_ms"] = latency
            c["status"] = "done"
        except Exception as e:
            c["status"] = "error"
            c["error"] = str(e)
        finally:
            return c

    max_workers = config.model.parallel_requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(work, c) for c in pending]
        for fut in concurrent.futures.as_completed(futures):
            _ = fut.result()
            pbar.update(1)
            _save_manifest(manifest_path, manifest, fs_provider)

    pbar.close()
    _save_manifest(manifest_path, manifest, fs_provider)