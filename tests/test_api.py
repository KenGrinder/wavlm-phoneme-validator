"""Route-level tests with a stubbed scorer.

The model is replaced rather than loaded: these assert the contract the caller depends
on -- request validation, how unmapped phones surface, and the shape of the response --
none of which need 1.2 GB of weights to be wrong.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import audio, runtime
from app.main import app

AUDIO = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


class StubAssessor:
    """Returns one scored phone per phone it was handed, so counts are checkable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def assess_phones(self, path, phone_info, text=None):
        self.calls.append(("assess_phones", [p["phone"] for p in phone_info]))
        phonemes = [
            {
                "phone": p["phone"],
                "word": p["word"],
                "word_idx": p["word_idx"],
                "score": 90.0,
                "gop": 1.0,
                "pherr_prob": 0.1,
                "error": False,
                "n_frames": 10,
                "start_ms": 20.0 * i,
                "end_ms": 20.0 * (i + 1),
            }
            for i, p in enumerate(phone_info)
        ]
        return {
            "text": text,
            "overall_score": 90.0,
            "n_phonemes": len(phonemes),
            "n_errors": 0,
            "error_rate": 0.0,
            "unmapped": [],
            "phonemes": phonemes,
            "words": [],
        }

    def assess(self, path, text):
        self.calls.append(("assess", text))
        return self.assess_phones(path, [{"phone": "k", "word": text, "word_idx": 0}], text)


@pytest.fixture
def client(monkeypatch):
    stub = StubAssessor()
    monkeypatch.setattr(runtime, "PRELOAD", False)
    monkeypatch.setattr(audio, "run", lambda assessor, path, entries, text:
                        assessor.assess(path, text) if entries is None else
                        assessor.assess_phones(path, entries, text))
    monkeypatch.setattr(runtime.STATE, "load", lambda: stub)
    monkeypatch.setattr(runtime.STATE, "device", "cpu")
    monkeypatch.setattr(runtime, "API_KEY", "")
    with TestClient(app) as c:
        c.stub = stub  # type: ignore[attr-defined]
        yield c


def post(client, **fields):
    return client.post(
        "/api/v1/assess",
        files={"file": ("clip.wav", AUDIO, "audio/wav")},
        data=fields,
    )


# ------------------------------------------------------------------- system


def test_health_is_liveness_only(client):
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_ready_is_503_until_the_models_are_loaded(client):
    """A container that reported ready during a 1.3 GB download would be handed work."""
    response = client.get("/api/v1/ready")
    assert response.status_code == 503
    assert response.json()["status"] in {"lazy", "warming", "error"}


def test_discovery_lists_the_endpoints(client):
    body = client.get("/api/v1").json()
    assert body["assess"] == "/api/v1/assess"


def test_info_reports_the_pinned_model(client):
    model = client.get("/api/v1/info").json()["model"]
    assert model["repo"] == "Jianshu001/wavlm-phoneme-scorer"
    assert len(model["revision"]) == 40
    assert len(model["checkpoint_sha256"]) == 64


# --------------------------------------------------------------- validation


def test_exactly_one_expectation_is_required(client):
    assert post(client).status_code == 400
    assert post(client, expected_ipa="kaet", text="cat").status_code == 400


def test_empty_upload_is_rejected(client):
    response = client.post(
        "/api/v1/assess",
        files={"file": ("clip.wav", b"", "audio/wav")},
        data={"expected_ipa": "k"},
    )
    assert response.status_code == 400


def test_threshold_is_bounded(client):
    assert post(client, expected_ipa="k", pherr_threshold=1.5).status_code == 400


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(runtime, "API_KEY", "s3cret")
    assert post(client, expected_ipa="k").status_code == 401
    ok = client.post(
        "/api/v1/assess",
        files={"file": ("clip.wav", AUDIO, "audio/wav")},
        data={"expected_ipa": "k"},
        headers={"X-API-Key": "s3cret"},
    )
    assert ok.status_code == 200


# ------------------------------------------------------------------ scoring


def test_expected_ipa_reaches_the_scorer_as_phones(client):
    body = post(client, expected_ipa="k æ t").json()
    assert client.stub.calls == [("assess_phones", ["k", "ae", "t"])]
    assert body["status"] == "ok"
    assert body["expected"]["source"] == "caller_ipa"
    assert [p["phone"] for p in body["phonemes"]] == ["k", "ae", "t"]


def test_each_scored_phone_carries_the_ipa_it_came_from(client):
    """So a reviewer is told /ae/, the symbol they wrote, not the scorer's `ae`."""
    body = post(client, expected_ipa="k æ t").json()
    assert [p["ipa"] for p in body["phonemes"]] == ["k", "æ", "t"]


def test_unmappable_phone_degrades_the_response_instead_of_shortening_it(client):
    """The failure that would corrupt a baseline invisibly, asserted directly."""
    body = post(client, expected_ipa="k ☃ t").json()
    assert body["status"] == "degraded"
    assert [u["ipa"] for u in body["unmapped"]] == ["☃"]
    assert client.stub.calls == []
    assert body["phonemes"] == []


def test_nothing_mappable_is_an_error_not_a_pass(client):
    response = post(client, expected_ipa="☃ ☄")
    assert response.status_code == 422


def test_text_path_uses_the_models_own_g2p(client):
    body = post(client, text="cat").json()
    assert client.stub.calls[0][0] == "assess"
    assert body["expected"]["source"] == "model_g2p"


def test_arpabet_path_strips_stress(client):
    post(client, expected_arpabet="K AE1 T")
    assert client.stub.calls == [("assess_phones", ["k", "ae", "t"])]


def test_response_records_the_model_revision(client):
    """A score is only comparable to a score from the same weights."""
    body = post(client, expected_ipa="k").json()
    assert body["model"]["revision"] == runtime.MODEL_REVISION
    assert body["device"] == "cpu"


def test_request_threshold_changes_all_error_summaries(client):
    body = post(client, expected_ipa="k æ t", pherr_threshold=0.05).json()
    assert body["n_errors"] == 3
    assert body["error_rate"] == 100
    assert all(p["error"] for p in body["phonemes"])
    assert body["words"][0]["n_errors"] == 3
    assert client.stub.__dict__.get("pherr_threshold") is None


def test_empty_model_output_is_not_success(client, monkeypatch):
    monkeypatch.setattr(client.stub, "assess_phones", lambda *args: {"phonemes": []})
    assert post(client, expected_ipa="k").status_code == 422


def test_whitespace_input_does_not_override_another_target(client):
    assert post(client, expected_ipa=" ", expected_arpabet="K").status_code == 200


def test_long_expectation_is_rejected_before_inference(client):
    assert post(client, expected_ipa="k" * 2001).status_code == 413
    assert client.stub.calls == []
