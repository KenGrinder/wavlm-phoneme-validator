"""Response models. These exist so /docs describes the contract the caller depends on."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PhonemeScore(BaseModel):
    phone: str = Field(description="Phone in the scorer's ARPABET-ish alphabet, lowercase.")
    ipa: str | None = Field(default=None, description="The IPA symbol this came from.")
    word: str | None = None
    word_idx: int = 0
    score: float = Field(description="0-100. Higher is a better-produced phone.")
    gop: float = Field(description="Goodness of pronunciation from the CTC posteriors.")
    pherr_prob: float = Field(description="0-1 probability this phone is mispronounced.")
    error: bool = Field(description="pherr_prob >= the request's threshold.")
    n_frames: int = 0
    start_ms: float | None = Field(
        default=None, description="Where in the clip this phone was aligned."
    )
    end_ms: float | None = None


class WordScore(BaseModel):
    word: str
    score: float
    n_errors: int
    n_phonemes: int
    has_error: bool
    phonemes: list[PhonemeScore] = []


class UnmappedPhone(BaseModel):
    position: int
    ipa: str
    word: str | None = None


class ExpectedPronunciation(BaseModel):
    source: str = Field(description="caller_ipa | caller_arpabet | model_g2p")
    ipa: list[str] = []
    phones: list[str] = []


class ModelPins(BaseModel):
    repo: str
    revision: str
    checkpoint_sha256: str
    ctc: str
    backbone: str
    ctc_revision: str
    backbone_revision: str
    wrapper_version: str


class AssessResponse(BaseModel):
    status: str = Field(
        description=(
            "ok when every expected phone was scored; degraded when one or more could "
            "not be represented in the CTC vocabulary and appear in `unmapped`."
        )
    )
    text: str | None = None
    overall_score: float
    n_phonemes: int
    n_errors: int
    error_rate: float
    pherr_threshold: float
    expected: ExpectedPronunciation
    unmapped: list[UnmappedPhone] = []
    phonemes: list[PhonemeScore] = []
    words: list[WordScore] = []
    duration_ms: float | None = None
    elapsed_ms: float
    device: str
    model: ModelPins


class Health(BaseModel):
    status: str


class Ready(BaseModel):
    status: str = Field(description="lazy | warming | ready | error")
    detail: str | None = None


class Limits(BaseModel):
    max_upload_mb: int
    max_concurrency: int
    default_pherr_threshold: float
    max_audio_seconds: int = 30
    max_text_length: int = 2000
    max_phones: int = 400


class Info(BaseModel):
    name: str
    version: str
    api_version: str
    device: str
    gpu: str | None = None
    models: Ready
    authentication_required: bool
    model: ModelPins
    limits: Limits
