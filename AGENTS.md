# AGENTS.md

A FastAPI wrapper around a vendored phoneme scorer. Small, but three things about it are
easy to break without noticing.

## Commands

```bash
pip install -r requirements-test.txt fastapi python-multipart
python -m ruff check app tests scripts
python -m pytest
python scripts/validate_templates.py
docker compose -f compose.cpu.yaml up --build
```

The tests deliberately do not need torch. Do not "fix" that by importing the vendored
pipeline in `app/phones.py` — see the docstring on `_load_arpabet_table`.

## Three things not to break

**1. The service does not decide what the correct pronunciation is.** It scores audio
against a target the caller supplies. If you find yourself adding a code path where the
model's own G2P overrides `expected_ipa`, stop: that is the failure this whole wrapper
exists to prevent.

**2. A phone that cannot be scored must be reported, never dropped.** Upstream removes
unmappable phones from the expected sequence, so a clip with an unrepresentable target
scores as a clean pass over the phones that happened to survive. `split_expected` and
the `unmapped` field are the guard. `test_unmappable_phone_degrades_the_response_instead_of_shortening_it`
is the test; if it starts failing, the bug is real, not the test.

**3. The CUDA pin is load-bearing.** `Dockerfile` targets Pascal (`sm_61`). cu128+ torch
wheels have no Pascal kernels and fail *quietly*. Do not bump the base image or
`TORCH_VERSION` without a run on the target card.

## Vendored upstream

`app/vendor/pipeline_v2.py` is pinned and patched; `patches/` holds the diff and
`app/vendor/UPSTREAM.md` explains each hunk. When bumping the revision, update
`MODEL_REVISION` and `CHECKPOINT_SHA256` in `app/runtime.py` in the same commit — scores
are not comparable across revisions, and the revision is recorded with every result so
that is visible rather than silent.

Ruff is configured to skip `app/vendor`. Reformatting it would bury the patch.

## Thresholds

`0.70` is upstream's calibration on children's speech. It is a default, not a finding.
Anything that turns a score into a pass/fail gate belongs in the caller, calibrated
against clips a human has already judged.
