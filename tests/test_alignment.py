from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from app.vendor.pipeline_v2 import PronunciationAssessorV2  # noqa: E402


def test_impossible_repeated_phone_alignment_returns_no_path():
    emissions = torch.log_softmax(torch.tensor([[0., 1.], [0., 1.]]), dim=-1)
    # Two identical CTC targets require an intervening blank (at least 3 frames).
    assert PronunciationAssessorV2._viterbi_align(emissions, [1, 1], 0) == []


def test_repeated_phone_alignment_retains_both_occurrences():
    emissions = torch.log_softmax(torch.tensor([[0., 9.], [9., 0.], [0., 9.]]), dim=-1)
    path = PronunciationAssessorV2._viterbi_align(emissions, [1, 1], 0)
    assert [token for _, token in path] == [1, 0, 1]
