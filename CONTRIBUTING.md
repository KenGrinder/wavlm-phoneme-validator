# Contributing

## Before a pull request

```bash
python -m ruff check app tests scripts
python -m pytest
python scripts/validate_templates.py
```

CI runs those, builds the CPU image, runs the suite *inside* it (where torch is present,
so the vendored-table check stops skipping), and smoke-tests the endpoints.

## Changing the vendored pipeline

Edit `app/vendor/pipeline_v2.py`, then regenerate the patch so the two stay in step:

```bash
curl -sSL -o /tmp/pristine.py \
  "https://huggingface.co/Jianshu001/wavlm-phoneme-scorer/resolve/<rev>/pipeline_v2.py"
diff -u /tmp/pristine.py app/vendor/pipeline_v2.py > patches/0001-assess-explicit-phones.patch
```

Keep the patch small and each hunk explainable in a sentence. The next upstream bump has
to be re-applied by hand, and a large diff is what makes that a rewrite instead of a
merge.

## Releasing

Tag `vX.Y.Z`. The release workflow builds and pushes `:gpu` and `:cpu`, plus
version-suffixed tags for pinning. Bump `<Changes>` in the Unraid template in the same
commit as the tag.
