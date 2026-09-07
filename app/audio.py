"""Bound decoded audio before it reaches the GPU."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

MAX_SECONDS = 30


def decode(path: str) -> tuple[str, float]:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is not installed")
    output = path + ".decoded.wav"
    try:
        subprocess.run(  # noqa: S603 - fixed decoder arguments and generated temp paths
            [executable, "-nostdin", "-v", "error", "-y", "-i", path,
             "-t", str(MAX_SECONDS + 1), "-ac", "1", "-ar", "16000",
             "-threads", "1", "-c:a", "pcm_s16le", output],
            check=True, timeout=30, capture_output=True,
        )
        with wave.open(output, "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
        if not 0.025 <= duration <= MAX_SECONDS:
            raise ValueError(f"audio must be between 25 ms and {MAX_SECONDS} seconds")
        return output, duration * 1000
    except (subprocess.SubprocessError, wave.Error) as exc:
        Path(output).unlink(missing_ok=True)
        raise ValueError("audio could not be decoded within the time limit") from exc
    except Exception:
        Path(output).unlink(missing_ok=True)
        raise


def run(assessor, path: str, entries, text):
    decoded, duration = decode(path)
    try:
        if entries is None:
            result = assessor.assess(decoded, text)
        else:
            result = assessor.assess_phones(decoded, entries, text)
        result["duration_ms"] = duration
        return result
    finally:
        Path(decoded).unlink(missing_ok=True)
