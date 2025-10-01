import concurrent.futures
import json
import os
import time
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI


TRANSIENT_ERRORS = ("rate_limit_exceeded", "server_error", "temporarily_unavailable")


def _save_manifest(manifest_path: Path, manifest: dict):
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _transcribe_one(client: OpenAI, chunk: dict, config):
    start = time.time()
    model = config.model.model
    response_format = config.model.response_format
    prompt = config.model.prompt

    with open(chunk["file"], "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format=response_format,
            prompt=prompt if prompt else None,
        )

    # Normalize to dict
    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
    else:
        try:
            data = json.loads(str(resp))
        except Exception:
            data = {"text": getattr(resp, "text", str(resp))}

    text = data.get("text") if isinstance(data, dict) else None
    latency_ms = int((time.time() - start) * 1000)
    return text, latency_ms


def _retrying_transcribe(client: OpenAI, chunk: dict, config):
    max_retries = config.model.max_retries
    backoff = config.model.backoff_base_ms

    for attempt in range(max_retries + 1):
        try:
            return _transcribe_one(client, chunk, config)
        except Exception as e:
            msg = str(e).lower()
            if attempt < max_retries and any(t in msg for t in TRANSIENT_ERRORS):
                delay = (2 ** attempt) * backoff / 1000.0
                time.sleep(delay)
                continue
            raise


def transcribe_manifest(manifest_path: Path, config):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    pending = [c for c in manifest["chunks"] if c.get("status") == "pending"]
    pbar = tqdm(total=len(pending), desc="Transcribing", unit="chunk")

    def work(c):
        try:
            text, latency = _retrying_transcribe(client, c, config)
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
            _save_manifest(manifest_path, manifest)

    pbar.close()
    _save_manifest(manifest_path, manifest)