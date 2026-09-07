#!/usr/bin/env bash
set -euo pipefail
image="$1"
config=$(mktemp -d)
name="wavlm-ci-${RANDOM}"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true; sudo rm -rf "$config"' EXIT
sudo chown 99:100 "$config"
docker run -d --name "$name" --user 99:100 --cap-drop=ALL \
  --security-opt=no-new-privileges -p 127.0.0.1::8000 \
  -v "$config:/config" -e WAVLM_PRELOAD_MODELS=0 "$image"
port=$(docker port "$name" 8000/tcp | awk -F: '{print $NF}')
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$port/health"
test "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/ready")" = 503
test "$(docker exec "$name" id -u)" = 99
docker exec "$name" sh -c 'test -w /config/huggingface && test -w /config/torch'
