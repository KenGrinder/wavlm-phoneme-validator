# WavLM Phoneme Validator

Phoneme-level pronunciation scoring as a self-hosted REST API, built on
[`Jianshu001/wavlm-phoneme-scorer`](https://huggingface.co/Jianshu001/wavlm-phoneme-scorer)
— a fine-tuned WavLM-Large backbone with an MLP scoring head, trained on 53,926
professionally labelled phonemes from 11,601 recordings of children learning English.

## Why this rather than a speech recogniser

A recogniser answers **"what did you hear?"**. This answers **"how well was *this
specific expected phone* produced?"** — forced alignment against a pronunciation you
supply, then a score and an error probability for each phone in it.

The two services provide complementary evidence. Whether this scorer helps on
isolated synthetic phones must be established with current human-labelled clips.
Its wav2vec2 CTC aligner can share failure modes with another wav2vec2 recogniser;
agreement raises review priority but does not prove a pronunciation is wrong.

## The rule this service is built around

**It never decides what the correct pronunciation is.** You supply the target; it only
judges whether the audio matched it. Upstream's `assess(audio, text)` runs its own G2P,
which means it can quietly grade a recording against a different pronunciation than the
one your synthesizer was handed. `expected_ipa` exists to close that gap, and it is the
path you should use in production.

## Quick start

```bash
docker compose -f compose.gpu.yaml up -d --build   # or compose.cpu.yaml
```

```bash
curl -X POST http://localhost:8005/api/v1/assess \
  -F "file=@cat.wav" \
  -F "expected_ipa=k æ t"
```

```json
{
  "status": "ok",
  "overall_score": 95.8,
  "n_phonemes": 3,
  "n_errors": 0,
  "expected": { "source": "caller_ipa", "ipa": ["k", "æ", "t"], "phones": ["k", "ae", "t"] },
  "unmapped": [],
  "phonemes": [
    { "phone": "ae", "ipa": "æ", "score": 93.1, "gop": -1.2, "pherr_prob": 0.10,
      "error": false, "start_ms": 120.0, "end_ms": 340.0 }
  ],
  "model": { "repo": "Jianshu001/wavlm-phoneme-scorer", "revision": "19f6b967…" },
  "device": "cuda"
}
```

The first start downloads several gigabytes into `/config`. `/api/v1/health` answers
immediately; `/api/v1/ready` answers `503` until the weights are loaded. Poll readiness,
not health.

## API

| | |
|---|---|
| `POST /api/v1/assess` | score a recording — see below |
| `GET /api/v1/health` | liveness; the process is up |
| `GET /api/v1/ready` | `lazy` / `warming` / `ready` / `error`, `503` unless ready |
| `GET /api/v1/info` | device, GPU, model pins, limits |
| `GET /api/v1` | discovery document |
| `GET /docs` | Swagger UI |

`POST /api/v1/assess` is `multipart/form-data` with `file` plus **exactly one**
expectation:

| field | use |
|---|---|
| `expected_ipa` | space-separated IPA — **the production path** |
| `expected_arpabet` | space-separated ARPABET; stress digits (`AE1`) are stripped |
| `text` | reference text, phonemised by the model's own `g2p_en` — convenience only |

Optional: `words` (space-separated, to group phones by word) and `pherr_threshold`
(0–1, default `0.70`).

### Two response details worth knowing

**Stress and length marks are ignored.** `ˈ`, `ˌ` and `ː` tell a synthesizer *how* to say
a phone; they do not change *which* phone is expected. `æː` and `æ` are the same target
here. Send whatever string you gave the synthesizer — it will be read correctly.

**`status: "degraded"` means something was not scored.** Any phone the CTC vocabulary
cannot represent is listed in `unmapped` rather than dropped. Upstream deletes those
silently, which produces a *shorter* expected sequence and a result that looks like a
clean pass. Treat `degraded` as "this clip was not fully checked".

## Configuration

| variable | default | |
|---|---|---|
| `WAVLM_DEVICE` | auto | `cuda` / `cpu` |
| `WAVLM_PRELOAD_MODELS` | `1` | warm at startup instead of on first request |
| `WAVLM_MAX_CONCURRENCY` | `1` | see the VRAM note below |
| `WAVLM_MAX_UPLOAD_MB` | `25` | |
| `WAVLM_PHERR_THRESHOLD` | `0.70` | upstream's calibration — see *Calibrate before trusting* |
| `WAVLM_API_KEY` | — | optional; `X-API-Key` or `Authorization: Bearer` |
| `WAVLM_CORS_ORIGINS` | — | comma-separated browser origins |
| `PORT` | `8000` | |

## Unraid

Add the template URL under **Docker → Add Container → Template**:

```
https://raw.githubusercontent.com/KenGrinder/wavlm-phoneme-validator/main/templates/wavlm-phoneme-validator.xml
```

It needs the NVIDIA Driver plugin and a working container runtime, and it maps host port
`8005` (chosen to sit clear of OpenPronounce on `8004` and Kokoro on `8880`).

## Hardware notes

The target server uses a GTX 1080 Ti (Pascal, sm_61), shared with Kokoro and
OpenPronounce. This image pins CUDA 12.6 and torch 2.7.1, a combination verified
by actual inference on that card. Check kernel support and run a real assessment
before changing either dependency; a driver's reported CUDA version alone does
not establish compatibility. FP32 is the default; benchmark before changing precision.

On 2026-09-07 total GPU usage rose from 5429 to about 7981 MiB after loading this
service alongside both existing services. This is a shared-card snapshot, not a
peak-memory guarantee. Keep concurrency at 1 and measure long-input peaks before
raising it. CPU mode is available; its latency has not been benchmarked here.

## Calibrate before trusting the numbers

The published threshold of `0.70` was calibrated on **children reading words and
sentences**. Two things about that are worth taking seriously before wiring this to a
gate:

1. **Synthesized speech is a different distribution** from a child at a microphone.
2. **An isolated phone is a different task again.** Forced-aligning a single phone over a
   300 ms clip is out of distribution for both the CTC aligner and the scorer. That is
   the case where this service is most attractive *and* least proven.

Score a set of clips you have already judged by ear, look at whether the known-bad ones
separate from the known-good, and set your own thresholds from that. If they do not
separate, the honest conclusion is that this model does not help on that asset class —
not that the clips were fine.

## Development

```bash
pip install -r requirements-test.txt fastapi python-multipart
python -m ruff check app tests scripts
python -m pytest
python scripts/validate_templates.py
```

The tests run without torch: the scorer is stubbed and the phone tables are parsed out of
the vendored pipeline rather than imported. `test_table_matches_import` asserts the parse
agrees with the real module, and CI runs the same suite inside the built image where
torch is present.

## Upstream

`app/vendor/pipeline_v2.py` is vendored at a pinned revision and carries one patch. See
[`app/vendor/UPSTREAM.md`](app/vendor/UPSTREAM.md) for what changed and why, and how to
re-apply it after an upstream bump.

## Licence

MIT, as is the upstream model.

## 1.0.2 phone coverage

Two classes of target were coming back unscored rather than scored badly, found on a
real 1457-clip catalogue where they accounted for all 43 incomplete assessments.

The alveolar tap is now scored as the /t/ it flaps. American English flaps /t/ between
vowels, so espeak phonemises a word like "thirty" with a tap and the scorer has no tap
of its own; every clip containing that word was reported degraded.

Clusters that are not phones -- `tr`, `dr`, `dz`, `ts`, `ar`, `ir`, `ur` -- are
decomposed by the tokenizer. Upstream lists them as single ARPABET keys, but the CTC
vocabulary has no symbol for any of them, so alignment failed and *neither* half was
scored. They remain in the lookup, so an explicit ARPABET `TR` still resolves.

This changes coverage, not accuracy. No clip scores better; 43 of them move from
"not checked" to "checked". `wrapper_version` is reported with every result, so scores
from 1.0.1 and 1.0.2 are distinguishable rather than silently pooled.

## 1.0.1 deployment and scoring corrections

The service runs directly as UID 99 / GID 100, including with `--cap-drop=ALL` and
`--security-opt=no-new-privileges`. Create the host config directory with that owner
before starting, or use `--user UID:GID` with a directory owned by that identity.
It no longer changes accounts or ownership at startup. CPU/GPU release smoke tests
exercise an empty bind mount with these same restrictions before publishing.

All three model revisions are fixed and reported in `/api/v1/info`. The checkpoint
hash is verified on each process start. Input audio is limited to 30 seconds and
25 MB; expectation strings to 2000 characters and 400 phones. An unsupported target
returns no partial scores. A failed/incomplete assessment is never a clean pass.

WavLM uses a wav2vec2 CTC aligner. Its errors may correlate with other wav2vec2
validators. It remains advisory for synthetic isolated phones until tested on
current human-labelled recordings. Agreement is a review priority, not a substitute
for a human pronunciation judgement. Docker health reports liveness only; clients
must also check readiness. Docker by itself does not restart unhealthy containers.
