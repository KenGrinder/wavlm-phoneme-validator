"""Model lifecycle: pinning, lazy loading, a background warm, and a concurrency gate."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import threading
import time

logger = logging.getLogger("wavlm")

WRAPPER_VERSION = "1.0.2"

# Pinned deliberately. A score is only comparable to another score from the same weights,
# so the revision travels with every result and a bump is a visible, reviewed change --
# never something that happens because a container restarted. See app/vendor/UPSTREAM.md.
MODEL_REPO = "Jianshu001/wavlm-phoneme-scorer"
MODEL_REVISION = "19f6b9675d02a61dcd5b2aec0fdcc4364c1fc9da"
CHECKPOINT_FILENAME = "wavlm_finetuned.pt"
CHECKPOINT_SHA256 = "7b9485b679d9a1219ac7dbef197b5185ec16e7909632b082b1f0576a963e0040"
CTC_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
BACKBONE_MODEL = "microsoft/wavlm-large"
CTC_REVISION = "2c733782da5604684829819a5eb744c193fe9398"
BACKBONE_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


MAX_CONCURRENCY = max(1, _env_int("WAVLM_MAX_CONCURRENCY", 1))
MAX_UPLOAD_MB = max(1, _env_int("WAVLM_MAX_UPLOAD_MB", 25))
PHERR_THRESHOLD = float(os.getenv("WAVLM_PHERR_THRESHOLD", "0.70"))
PRELOAD = _env_flag("WAVLM_PRELOAD_MODELS", True)
API_KEY = (os.getenv("WAVLM_API_KEY") or "").strip()


class ModelState:
    """Tracks load status so /ready can answer without blocking on a 1.2 GB download."""

    def __init__(self) -> None:
        self.status = "lazy"          # lazy | warming | ready | error
        self.detail: str | None = None
        self.device: str = "unknown"
        self.gpu: str | None = None
        self.loaded_at: float | None = None
        self._assessor = None
        self._lock = threading.Lock()

    # -- device ----------------------------------------------------------------

    @staticmethod
    def _pick_device():
        import torch

        wanted = (os.getenv("WAVLM_DEVICE") or "").strip().lower()
        if wanted:
            return torch.device(wanted)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- checkpoint ------------------------------------------------------------

    @staticmethod
    def _resolve_checkpoint() -> str:
        """Download at the pinned revision and verify the bytes we got.

        The hash check is not ceremony: the checkpoint is a pickled `.pt` loaded with
        `weights_only=False`. Re-hash the bytes on each process start; a persistent
        marker cannot prove that a cached file has not changed.
        """
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=CHECKPOINT_FILENAME,
            revision=MODEL_REVISION,
        )

        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(8 << 20), b""):
                digest.update(block)
        got = digest.hexdigest()
        if got != CHECKPOINT_SHA256:
            raise RuntimeError(
                f"checkpoint sha256 mismatch: expected {CHECKPOINT_SHA256}, got {got}"
            )
        return path

    # -- load ------------------------------------------------------------------

    def load(self):
        """Blocking load. Safe to call from several threads; only one does the work."""
        if self._assessor is not None:
            return self._assessor
        with self._lock:
            if self._assessor is not None:
                return self._assessor
            self.status = "warming"
            self.detail = None
            started = time.monotonic()
            try:
                import torch

                from .vendor.pipeline_v2 import PronunciationAssessorV2

                device = self._pick_device()
                self.device = str(device)
                if device.type == "cuda":
                    self.gpu = torch.cuda.get_device_name(device)

                assessor = PronunciationAssessorV2(
                    checkpoint_path=self._resolve_checkpoint(),
                    device=device,
                    pherr_threshold=PHERR_THRESHOLD,
                    ctc_revision=CTC_REVISION,
                    backbone_revision=BACKBONE_REVISION,
                )
                assessor._load_models()
                self._assessor = assessor
                self.status = "ready"
                self.loaded_at = time.time()
                logger.info(
                    "models ready on %s in %.1fs", self.device, time.monotonic() - started
                )
            except Exception as exc:  # noqa: BLE001 - surfaced through /ready
                self.status = "error"
                self.detail = f"{type(exc).__name__}: {exc}"
                logger.exception("model load failed")
                raise
            return self._assessor

    @property
    def assessor(self):
        return self._assessor

    def snapshot(self) -> dict:
        return {"status": self.status, "detail": self.detail}


STATE = ModelState()

# One clip at a time by default. The target box is a GTX 1080 Ti with ~5.8 GB free after
# Kokoro and OpenPronounce, and two concurrent WavLM-Large forward passes will not fit.
GATE = asyncio.Semaphore(MAX_CONCURRENCY)


async def assess(fn, *args, **kwargs):
    """Run a blocking scorer call off the event loop, behind the concurrency gate."""
    async with GATE:
        task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # A cancelled HTTP request cannot release the GPU slot or delete its
            # input while the blocking worker is still using them.
            try:
                await task
            finally:
                raise


def warm_in_background() -> None:
    if not PRELOAD:
        return
    threading.Thread(target=_warm, name="wavlm-warm", daemon=True).start()


def _warm() -> None:
    # STATE.load() already logs the traceback and records the failure for /ready, so
    # there is nothing left to do here but let the warm thread exit quietly.
    with contextlib.suppress(Exception):
        STATE.load()


def model_info() -> dict:
    return {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "ctc": CTC_MODEL,
        "backbone": BACKBONE_MODEL,
        "ctc_revision": CTC_REVISION,
        "backbone_revision": BACKBONE_REVISION,
        "wrapper_version": WRAPPER_VERSION,
    }
