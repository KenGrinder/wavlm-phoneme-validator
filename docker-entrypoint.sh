#!/usr/bin/env bash
set -euo pipefail

# Unraid's appdata is owned by nobody:users (99:100). Running as root would leave a
# 1.3 GB model cache root-owned inside a share the array tools then cannot manage.
PUID="${PUID:-99}"
PGID="${PGID:-100}"
UMASK="${UMASK:-022}"

if ! [[ "${PUID}" =~ ^[0-9]+$ && "${PGID}" =~ ^[0-9]+$ ]]; then
    echo "PUID and PGID must be numeric" >&2
    exit 2
fi

if ! getent group appgroup >/dev/null 2>&1; then
    groupadd --non-unique --gid "${PGID}" appgroup
fi
if ! id appuser >/dev/null 2>&1; then
    useradd \
        --system \
        --non-unique \
        --uid "${PUID}" \
        --gid appgroup \
        --home-dir /config \
        --shell /bin/bash \
        appuser
fi

mkdir -p /config/huggingface /config/torch /config/cache
# Only the directories we create, not -R over the whole volume: a recursive chown across
# a 1.3 GB model cache on spinning disks adds minutes to every restart for no benefit.
chown "${PUID}:${PGID}" /config /config/huggingface /config/torch /config/cache
umask "${UMASK}"

export HOME=/config
export HF_HOME="${HF_HOME:-/config/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/config/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/config/cache}"

exec gosu "${PUID}:${PGID}" "$@"
