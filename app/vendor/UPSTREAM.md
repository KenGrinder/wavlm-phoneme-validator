# Vendored upstream

`pipeline_v2.py` is vendored rather than pip-installed because the upstream model ships
its inference pipeline as a loose file on the Hugging Face Hub, with no package and no
release tags. Vendoring pins it; patching it in-tree keeps the diff reviewable.

| | |
|---|---|
| Source | <https://huggingface.co/Jianshu001/wavlm-phoneme-scorer> |
| License | MIT |
| Revision | `19f6b9675d02a61dcd5b2aec0fdcc4364c1fc9da` |
| File | `pipeline_v2.py`, 27586 bytes, git blob `a72e13a754998bc3060484ea8c72e3b69f62ba32` |
| Checkpoint | `wavlm_finetuned.pt`, 1265922672 bytes, sha256 `7b9485b679d9a1219ac7dbef197b5185ec16e7909632b082b1f0576a963e0040` |

The vendored copy is stored with LF line endings; the pristine download from the Hub has
the same bytes.

## Applied patches

`patches/0001-assess-explicit-phones.patch` — three changes:

1. **`assess_phones(audio_path, phone_info, text=None)`** splits out of `assess()`, so a
   caller that already has a canonical pronunciation can supply the expected phone
   sequence. Without this the model runs its own `g2p_en` and scores the audio against a
   different target than the one the synthesizer was handed — the exact failure the
   pronunciation-authority layer exists to prevent. `assess(audio, text)` is unchanged
   from the caller's point of view and now delegates.

2. **Unmapped phones are reported, not dropped.** Upstream's `_arpabet_to_model_idx`
   returns `-1` for anything absent from the CTC vocabulary, and `assess()` then removed
   it from both the index list and `valid_info` — so the model scored a *shorter sequence
   than requested* with no way for the caller to notice. Those now come back in
   `unmapped`. This matters here: the phone inventory includes `ɹː`, `æː`, `ʔ` and Misaki
   shorthand that will not map.

3. **Per-phone `start_ms` / `end_ms`,** derived from the forced-alignment frame span and a
   frame duration measured from the clip rather than assumed. A reviewer gets told which
   200 ms of a clip to listen to instead of replaying all of it.

## Re-applying after an upstream bump

```bash
curl -sSL -o app/vendor/pipeline_v2.py \
  "https://huggingface.co/Jianshu001/wavlm-phoneme-scorer/resolve/<new-rev>/pipeline_v2.py"
git apply patches/0001-assess-explicit-phones.patch   # resolve conflicts by hand
pytest                                                 # the tests cover all three changes
```

Update the revision and checkpoint hashes in this file and in `app/runtime.py`
(`MODEL_REVISION`, `CHECKPOINT_SHA256`) together — a score is not comparable across
revisions, so both are recorded with every result.
