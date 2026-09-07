"""Phone-alphabet normalisation.

The caller has a canonical pronunciation in IPA; the scorer works in the ARPABET-ish
keys of ``ARPABET_TO_IPA`` and can only align phones that exist in the CTC model's
vocabulary. This module is the single place those three alphabets meet, and the one
place a phone is allowed to be dropped -- loudly, into ``unmapped``.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from pathlib import Path

_VENDORED = Path(__file__).with_name("vendor") / "pipeline_v2.py"


def _load_arpabet_table() -> dict[str, list[str]]:
    """Read ARPABET_TO_IPA out of the vendored pipeline.

    Parsed from the source rather than imported, because importing it pulls in torch,
    torchaudio, transformers and g2p_en -- roughly 2 GB of wheels for one dict literal.
    Normalisation is where the subtle bugs live, so its tests should run anywhere. The
    parse is pinned to the vendored file, and `test_table_matches_import` asserts it
    agrees with the real import wherever torch is installed.
    """
    tree = ast.parse(_VENDORED.read_text(encoding="utf-8"), filename=str(_VENDORED))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ARPABET_TO_IPA" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"ARPABET_TO_IPA not found in {_VENDORED}")


ARPABET_TO_IPA = _load_arpabet_table()
ALL_PHONES = sorted(ARPABET_TO_IPA)

# Diacritics that describe *how* to say a phone, not *which* phone it is.
#
# Length marks are the load-bearing case. The calling app writes /ae:/ deliberately --
# a bare 280 ms vowel was reviewed as inaudible, and doubling it is what made the sound
# teachable. That is an instruction to the synthesizer about duration; the phone identity
# is still /ae/, and stripping the mark here is what lets the scorer find it.
_STRIP = str.maketrans(
    "",
    "",
    "\u02c8"  # PRIMARY STRESS
    "\u02cc"  # SECONDARY STRESS
    "\u02d0"  # LENGTH MARK
    "\u02d1"  # HALF-LENGTH
    "\u0361"  # COMBINING DOUBLE INVERTED BREVE (tie bar)
    "\u035c"  # COMBINING DOUBLE BREVE BELOW
    "\u0329"  # SYLLABIC
    "\u02b2"  # PALATALISED
    "'\u2019:",
)

# Separators only. Unsupported speech sounds, including glottal stops, remain
# visible as unmapped targets; the service must never silently remove them.
_DISCARD = {"-", "_", "|", "."}

# Notational variants that mean the same phone before we ever reach the lookup table.
_ALIASES = {
    "\u0261": "g",   # LATIN SMALL LETTER SCRIPT G -> g
    "\u02a4": "d\u0292",  # d3 ligature -> dz
    "\u02a7": "t\u0283",  # tS ligature -> tS
    "\u1d4a": "\u0259",   # superscript schwa
    "\u1d7b": "\u026a",   # barred small capital I -> I
}

# Kokoro/Misaki shorthand. Storage and display notation in the calling app; it must be
# projected to real IPA before anything acoustic sees it. Same table as the app's
# MISAKI_TO_IPA, kept here so the service is correct when driven by anything else.
MISAKI_TO_IPA = {
    "A": "e\u026a", "I": "a\u026a", "O": "o\u028a", "W": "a\u028a", "Y": "\u0254\u026a",
    "T": "t",
}


def _build_ipa_index() -> dict[str, str]:
    """Invert ARPABET_TO_IPA.

    Several IPA symbols appear under more than one ARPABET key (/6/ under both `aa` and
    `ao`, /U/ under both `uh` and `oo`). Primary variants win, so /A/ resolves to `ah`
    rather than `ax` -- deterministic, and matching the order upstream declares.
    """
    index: dict[str, str] = {}
    for arpa, variants in ARPABET_TO_IPA.items():
        if variants:
            index.setdefault(variants[0], arpa)
    for arpa, variants in ARPABET_TO_IPA.items():
        for ipa in variants:
            index.setdefault(ipa, arpa)
    # Reduced central vowels the table does not list explicitly.
    index.setdefault("\u0259", "ah")   # schwa
    index.setdefault("\u025c", "er")   # open-mid central unrounded
    return index


IPA_TO_PHONE = _build_ipa_index()

# Longest-match first, so /tS/ and /aI/ are one phone rather than two.
_IPA_SYMBOLS = sorted(IPA_TO_PHONE, key=len, reverse=True)


class PhoneError(ValueError):
    """The caller's expected pronunciation could not be read at all."""


def _clean(token: str) -> str:
    token = unicodedata.normalize("NFC", token)
    for src, dst in _ALIASES.items():
        token = token.replace(src, dst)
    for src, dst in MISAKI_TO_IPA.items():
        token = token.replace(src, dst)
    token = token.translate(_STRIP)
    return "".join(c for c in token if c not in _DISCARD).strip()


def tokenize_ipa(ipa: str) -> list[str]:
    """Split an IPA string into phones, longest match first.

    Accepts either a pre-tokenised, space-separated string (what the calling app sends,
    since it already tokenises for its own comparison) or a raw run of IPA.
    """
    out: list[str] = []
    for chunk in ipa.split():
        cleaned = _clean(chunk)
        i = 0
        while i < len(cleaned):
            for sym in _IPA_SYMBOLS:
                if sym and cleaned.startswith(sym, i):
                    out.append(sym)
                    i += len(sym)
                    break
            else:
                out.append(cleaned[i])
                i += 1
    return [t for t in out if t]


def phones_from_ipa(ipa: str, words: list[str] | None = None) -> list[dict]:
    """IPA -> ``[{phone, word, word_idx, ipa}]`` in the scorer's alphabet.

    A phone with no ARPABET equivalent is kept with ``phone=None`` so the caller can
    report it as unmapped rather than quietly shortening the expected sequence.
    """
    groups = ipa.split() if (words and len(ipa.split()) == len(words)) else None
    entries: list[dict] = []

    if groups and words:
        for word_idx, (chunk, word) in enumerate(zip(groups, words, strict=True)):
            for token in tokenize_ipa(chunk):
                entries.append(_entry(token, word, word_idx))
    else:
        word = words[0] if words else ""
        for token in tokenize_ipa(ipa):
            entries.append(_entry(token, word, 0))

    if not entries:
        raise PhoneError(f"no phones could be read from {ipa!r}")
    return entries


def _entry(token: str, word: str, word_idx: int) -> dict:
    return {
        "phone": IPA_TO_PHONE.get(token),
        "word": word,
        "word_idx": word_idx,
        "ipa": token,
    }


_STRESS_DIGIT = re.compile(r"\d")


def phones_from_arpabet(arpabet: str, words: list[str] | None = None) -> list[dict]:
    """ARPABET -> the same shape. Stress digits (``AE1``) are tolerated and stripped.

    Stress is kept out of the acoustic comparison on purpose: the scorer has no lexical
    stress encoding, so ``AE1`` and ``AE0`` are the same target to it. Storing stress is
    still worthwhile upstream of here -- it is what the synthesizer needs.
    """
    entries: list[dict] = []
    word = words[0] if words else ""
    for raw in arpabet.split():
        phone = _STRESS_DIGIT.sub("", raw).strip().lower()
        if not phone:
            continue
        entries.append({
            "phone": phone if phone in ALL_PHONES else None,
            "word": word,
            "word_idx": 0,
            "ipa": (ARPABET_TO_IPA.get(phone) or [raw])[0],
        })
    if not entries:
        raise PhoneError(f"no phones could be read from {arpabet!r}")
    return entries


def split_expected(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition into what the scorer can align and what it cannot."""
    usable = [e for e in entries if e["phone"]]
    unmapped = [
        {"position": i, "ipa": e["ipa"], "word": e["word"] or None}
        for i, e in enumerate(entries)
        if not e["phone"]
    ]
    return usable, unmapped
