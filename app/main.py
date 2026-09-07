"""WavLM Phoneme Validator -- a second, independent opinion on generated speech.

The service answers one question: *how well was this specific expected phone produced?*
It never decides what the expected phone should be. That belongs to the caller's
pronunciation-authority layer, and letting an acoustic model re-derive the target with
its own G2P is how a validator ends up grading audio against a pronunciation nobody
asked for.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import runtime
from .phones import PhoneError, phones_from_arpabet, phones_from_ipa, split_expected
from .schemas import AssessResponse, Health, Info, Ready

logging.basicConfig(
    level=os.getenv("WAVLM_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("wavlm")

API_PREFIX = "/api/v1"
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm", ".aac"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.warm_in_background()
    yield


app = FastAPI(
    title="WavLM Phoneme Validator",
    version=runtime.WRAPPER_VERSION,
    description=(
        "Self-hosted phoneme-level pronunciation scoring built on "
        "Jianshu001/wavlm-phoneme-scorer. Accepts an explicit expected pronunciation so "
        "audio is graded against the caller's canonical target, not the model's own G2P."
    ),
    lifespan=lifespan,
)

_origins = [o.strip() for o in (os.getenv("WAVLM_CORS_ORIGINS") or "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


def require_key(request: Request) -> None:
    """Optional shared secret, the same shape OpenPronounce uses so both are driven alike."""
    if not runtime.API_KEY:
        return
    supplied = request.headers.get("x-api-key") or ""
    if not supplied:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    if supplied != runtime.API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# ----------------------------------------------------------------------- system


@app.get(API_PREFIX, tags=["system"], summary="API discovery document")
def discovery() -> dict:
    return {
        "name": "WavLM Phoneme Validator",
        "version": runtime.WRAPPER_VERSION,
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "health": f"{API_PREFIX}/health",
        "ready": f"{API_PREFIX}/ready",
        "info": f"{API_PREFIX}/info",
        "assess": f"{API_PREFIX}/assess",
    }


@app.get(f"{API_PREFIX}/health", response_model=Health, tags=["system"])
@app.get("/health", response_model=Health, include_in_schema=False)
def health() -> Health:
    """Liveness only. It says the process is up, not that it can score anything."""
    return Health(status="ok")


@app.get(
    f"{API_PREFIX}/ready",
    response_model=Ready,
    tags=["system"],
    summary="Model readiness (503 until loaded)",
)
@app.get("/ready", response_model=Ready, include_in_schema=False)
def ready() -> JSONResponse:
    """Separate from /health because the first start downloads ~1.3 GB of weights.

    A container that reported healthy during that window would be handed traffic it
    cannot serve, so readiness is its own signal and answers 503 until the load finishes.
    """
    snapshot = runtime.STATE.snapshot()
    return JSONResponse(snapshot, status_code=200 if snapshot["status"] == "ready" else 503)


@app.get(f"{API_PREFIX}/info", response_model=Info, tags=["system"])
def info() -> Info:
    return Info(
        name="WavLM Phoneme Validator",
        version=runtime.WRAPPER_VERSION,
        api_version="v1",
        device=runtime.STATE.device,
        gpu=runtime.STATE.gpu,
        models=Ready(**runtime.STATE.snapshot()),
        authentication_required=bool(runtime.API_KEY),
        model=runtime.model_info(),
        limits={
            "max_upload_mb": runtime.MAX_UPLOAD_MB,
            "max_concurrency": runtime.MAX_CONCURRENCY,
            "default_pherr_threshold": runtime.PHERR_THRESHOLD,
        },
    )


# ----------------------------------------------------------------------- assess


@app.post(
    f"{API_PREFIX}/assess",
    response_model=AssessResponse,
    tags=["scoring"],
    dependencies=[Depends(require_key)],
    summary="Score a recording against an expected pronunciation",
)
async def assess(
    file: UploadFile = File(..., description="Audio in any format ffmpeg/torchaudio reads."),
    expected_ipa: str | None = Form(
        default=None,
        description=(
            "Space-separated IPA phones. The primary path: send the same string the "
            "synthesizer was given, so both sides grade against one target. Stress and "
            "length marks are tolerated and ignored -- they instruct the synthesizer, "
            "they do not change which phone is expected."
        ),
    ),
    expected_arpabet: str | None = Form(
        default=None,
        description="Space-separated ARPABET. Stress digits (AE1) are stripped.",
    ),
    text: str | None = Form(
        default=None,
        description=(
            "Reference text, phonemised by the model's own g2p_en. Convenience and "
            "parity only -- prefer expected_ipa when a canonical pronunciation exists."
        ),
    ),
    words: str | None = Form(
        default=None,
        description="Optional space-separated words, to group the phones by word.",
    ),
    pherr_threshold: float | None = Form(
        default=None,
        description="Override the error threshold (0-1). Default 0.70, as calibrated upstream.",
    ),
) -> AssessResponse:
    supplied = [f for f in (expected_ipa, expected_arpabet, text) if f and f.strip()]
    if len(supplied) != 1:
        raise HTTPException(
            status_code=400,
            detail="supply exactly one of expected_ipa, expected_arpabet or text",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(payload) > runtime.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"upload exceeds {runtime.MAX_UPLOAD_MB} MB"
        )

    threshold = pherr_threshold if pherr_threshold is not None else runtime.PHERR_THRESHOLD
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="pherr_threshold must be within 0-1")

    word_list = words.split() if words else None
    try:
        if expected_ipa:
            entries, source = phones_from_ipa(expected_ipa, word_list), "caller_ipa"
        elif expected_arpabet:
            entries, source = phones_from_arpabet(expected_arpabet, word_list), "caller_arpabet"
        else:
            entries, source = None, "model_g2p"
    except PhoneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        assessor = runtime.STATE.load()
    except Exception as exc:  # noqa: BLE001 - the reason is already on /ready
        raise HTTPException(status_code=503, detail=f"models unavailable: {exc}") from exc

    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".wav"

    tmp_path: str | None = None
    started = time.monotonic()
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(payload)

        if entries is None:
            usable, unmapped = [], []
            result = await runtime.assess(assessor.assess, tmp_path, text)
        else:
            usable, unmapped = split_expected(entries)
            if not usable:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "no expected phone could be mapped to the scorer's alphabet: "
                        + " ".join(u["ipa"] for u in unmapped)
                    ),
                )
            result = await runtime.assess(
                assessor.assess_phones, tmp_path, usable, text
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("assessment failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        if tmp_path:
            with suppress(OSError):
                os.remove(tmp_path)

    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])

    # Two sources of dropped phones: ones we refused to send, and ones the patched
    # pipeline could not place. Both are reported; neither is allowed to shorten the
    # expected sequence silently, because that would look like a clean pass.
    unmapped = unmapped + [
        {
            "position": u.get("position", -1),
            "ipa": str(u.get("phone", "?")),
            "word": u.get("word"),
        }
        for u in result.get("unmapped", [])
    ]

    expected_ipa_list = [e["ipa"] for e in usable]
    scored = result.get("phonemes", [])
    phonemes = [
        {**ph, "ipa": expected_ipa_list[i] if i < len(expected_ipa_list) else None}
        for i, ph in enumerate(scored)
    ]
    ends = [p["end_ms"] for p in phonemes if p.get("end_ms") is not None]

    return AssessResponse(
        status="degraded" if unmapped else "ok",
        text=result.get("text"),
        overall_score=result.get("overall_score", 0.0),
        n_phonemes=result.get("n_phonemes", len(phonemes)),
        n_errors=result.get("n_errors", 0),
        error_rate=result.get("error_rate", 0.0),
        pherr_threshold=threshold,
        expected={
            "source": source,
            "ipa": expected_ipa_list,
            "phones": [e["phone"] for e in usable] or [p["phone"] for p in scored],
        },
        unmapped=unmapped,
        phonemes=phonemes,
        words=result.get("words", []),
        duration_ms=max(ends) if ends else None,
        elapsed_ms=round((time.monotonic() - started) * 1000, 1),
        device=runtime.STATE.device,
        model=runtime.model_info(),
    )
