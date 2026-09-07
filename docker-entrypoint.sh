#!/usr/bin/env bash
set -euo pipefail
# Run as the volume owner; no account edits, chown, or setuid at runtime.
# The image defaults to Unraid nobody:users. Override with docker --user UID:GID.
umask "${UMASK:-022}"
mkdir -p /config/huggingface /config/torch /config/cache
exec "$@"
