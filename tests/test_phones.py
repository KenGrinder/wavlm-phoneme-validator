"""Normalisation tests.

These run without torch. That is deliberate -- the alphabet layer is where a wrong
answer is silent (a dropped phone looks exactly like a clean pass), so it needs tests
that anyone can run on any machine, not only where a 2 GB GPU stack is installed.

The IPA below is taken from the calling app's real catalogue, not invented.
"""

from __future__ import annotations

import pytest

from app.phones import (
    ARPABET_TO_IPA,
    IPA_TO_PHONE,
    PhoneError,
    phones_from_arpabet,
    phones_from_ipa,
    split_expected,
    tokenize_ipa,
)


def phones(ipa: str, words: list[str] | None = None) -> list[str]:
    usable, _ = split_expected(phones_from_ipa(ipa, words))
    return [e["phone"] for e in usable]


def unmapped(ipa: str) -> list[str]:
    _, dropped = split_expected(phones_from_ipa(ipa))
    return [e["ipa"] for e in dropped]


# ------------------------------------------------------------------ tokenising


@pytest.mark.parametrize(
    "ipa,expected",
    [
        ("k æ t", ["k", "ae", "t"]),          # already tokenised by the caller
        ("kæt", ["k", "ae", "t"]),            # or one run of IPA
        ("tʃ", ["ch"]),                       # affricate is ONE phone
        ("dʒ", ["jh"]),
        ("aɪ", ["ay"]),                       # so is a diphthong
        ("oʊ", ["ow"]),
        ("ɔɪ", ["oy"]),
    ],
)
def test_multi_character_phones_are_not_split(ipa, expected):
    assert phones(ipa) == expected


def test_clusters_are_two_phones():
    """/ks/ is two segments. Gluing it into one cost the calling app a false positive."""
    assert phones("ɛks") == ["eh", "k", "s"]


# ------------------------------------------------------------------ diacritics


def test_length_mark_is_duration_not_identity():
    """The app writes doubled vowels because a bare 280 ms /ae/ was inaudible.

    That is an instruction to the synthesizer. The phone being taught is still /ae/, and
    if the length mark reached the lookup the phone would come back unmapped and the
    clip would score as if nothing was expected at all.
    """
    assert phones("æː") == ["ae"]
    assert phones("ɹː") == ["r"]
    assert phones("æːː") == ["ae"]


def test_stress_marks_are_ignored():
    assert phones("ˈnoʊ") == ["n", "ow"]
    assert phones("ˌbənˈænə") == phones("bənænə")


def test_glottal_stop_is_reported_as_unmapped():
    """A glottal onset is a synthesis artefact, not a sound this app teaches."""
    assert phones("ʔæ") == ["ae"]
    assert unmapped("ʔæ") == ["ʔ"]


# -------------------------------------------------------------------- notation


def test_misaki_shorthand_is_projected_to_real_ipa():
    """/n'O/ renders as NOW and /n'ou/ renders as NO -- the app stores the former.

    The scorer must see real IPA or it would align /now/ against a vowel it does not
    have, so the shorthand is expanded before the lookup.
    """
    assert phones("nˈO") == ["n", "ow"]
    assert phones("ˈI") == ["ay"]
    assert phones("ˈA") == ["ey"]
    assert phones("ˈW") == ["aw"]


def test_ligatures_and_script_g_fold_to_their_canonical_form():
    assert phones("ʤ") == phones("dʒ") == ["jh"]
    assert phones("ʧ") == phones("tʃ") == ["ch"]
    assert phones("ɡ") == phones("g") == ["g"]


# ------------------------------------------------------------------- unmapped


def test_unknown_symbol_is_reported_never_dropped():
    """The one failure mode that would corrupt a baseline invisibly."""
    dropped = unmapped("k ☃ t")
    assert dropped == ["☃"]
    assert phones("k ☃ t") == ["k", "t"]


def test_unmapped_carries_its_position():
    _, dropped = split_expected(phones_from_ipa("k ☃ t"))
    assert dropped[0]["position"] == 1


def test_nothing_readable_raises():
    with pytest.raises(PhoneError):
        phones_from_ipa("   ")


# -------------------------------------------------------------------- ARPABET


def test_stress_digits_are_stripped():
    assert [e["phone"] for e in phones_from_arpabet("K AE1 T")] == ["k", "ae", "t"]
    assert [e["phone"] for e in phones_from_arpabet("K AE0 T")] == ["k", "ae", "t"]


def test_unknown_arpabet_is_unmapped_not_guessed():
    usable, dropped = split_expected(phones_from_arpabet("K ZZ T"))
    assert [e["phone"] for e in usable] == ["k", "t"]
    assert [e["ipa"] for e in dropped] == ["ZZ"]


# ------------------------------------------------------------------- grouping


def test_words_group_phones_when_the_counts_line_up():
    grouped = phones_from_ipa("kæt dɔɡ", ["cat", "dog"])
    assert [(e["word"], e["phone"]) for e in grouped] == [
        ("cat", "k"), ("cat", "ae"), ("cat", "t"),
        ("dog", "d"), ("dog", "ao"), ("dog", "g"),
    ]


def test_mismatched_word_count_falls_back_to_one_group():
    """Better one honest group than a silently wrong alignment of words to phones."""
    grouped = phones_from_ipa("k æ t", ["cat", "dog"])
    assert {e["word_idx"] for e in grouped} == {0}


# ----------------------------------------------------------- table provenance


def test_every_table_entry_resolves_to_a_known_phone():
    assert set(IPA_TO_PHONE.values()) <= set(ARPABET_TO_IPA)


def test_primary_variants_win():
    """/A/ is listed under both `ah` and `ax`; the resolution must be deterministic."""
    assert IPA_TO_PHONE["ʌ"] == "ah"
    assert IPA_TO_PHONE["ə"] == "ax"


def test_table_matches_import():
    """The parsed table must equal the one the scorer actually uses.

    Skipped where torch is absent, which is the whole point of parsing it -- but it runs
    in the image build, so a drift between the parse and the vendored module is caught.
    """
    vendored = pytest.importorskip("app.vendor.pipeline_v2")
    assert ARPABET_TO_IPA == vendored.ARPABET_TO_IPA
    assert sorted(ARPABET_TO_IPA) == vendored.ALL_PHONES


def test_tokenizer_is_total():
    """Anything in the table round-trips; nothing silently vanishes."""
    for ipa in IPA_TO_PHONE:
        assert tokenize_ipa(ipa), f"{ipa!r} tokenised to nothing"


# ------------------------------------------------- what the aligner can reach

def test_alveolar_tap_scores_as_the_t_it_flaps():
    """espeak phonemises "thirty" with a tap, and the scorer has no tap symbol.

    Left alone this is not a wrong score, it is NO score: the phone comes back
    unmapped and the clip is reported as not fully checked. Measured on the real
    catalogue, this one symbol accounted for 40 of 43 unscorable clips.
    """
    usable, dropped = split_expected(phones_from_ipa("θ ɝ ɾ i"))
    assert [e["phone"] for e in usable] == ["th", "er", "t", "iy"]
    assert dropped == []


@pytest.mark.parametrize(
    "ipa,expected",
    [
        ("tɹ", ["t", "r"]),      # praise.tried_another_way, praise.nice_try
        ("dɹ", ["d", "r"]),
        ("dz", ["d", "z"]),      # hint.countdown.2
        ("ts", ["t", "s"]),
        ("ɪɹ", ["ih", "r"]),
        ("ʊɹ", ["uh", "r"]),
    ],
)
def test_clusters_decompose_because_the_ctc_vocab_has_no_symbol_for_them(ipa, expected):
    """The upstream table lists these as single ARPABET keys; alignment cannot use them.

    Keeping them whole meant neither half was scored. Splitting them means both are.
    """
    assert phones(ipa) == expected


def test_real_phones_are_still_whole():
    """The decomposition must not reach the affricates and diphthongs beside them."""
    assert phones("tʃ") == ["ch"]
    assert phones("dʒ") == ["jh"]
    assert phones("aɪ") == ["ay"]
    assert phones("oʊ") == ["ow"]
