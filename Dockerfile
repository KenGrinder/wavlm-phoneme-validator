# CUDA 12.6 is the newest toolchain whose PyTorch wheels still ship sm_60/sm_61 kernels,
# which are what a GTX 1080 Ti executes. Do not bump to cu128/cu129 wheels: PyTorch
# dropped Maxwell/Pascal from those builds in 2.8, and the failure is quiet -- the
# container starts, reports cuda, and then either falls back to CPU or dies inside the
# first forward pass. The driver advertising "CUDA 13.0" is its ceiling, not a target.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ARG TORCH_VERSION=2.7.1
ARG TORCH_CUDA=cu126

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/srv" \
    HOME="/config" \
    HF_HOME="/config/huggingface" \
    TORCH_HOME="/config/torch" \
    XDG_CACHE_HOME="/config/cache" \
    NLTK_DATA="/usr/local/share/nltk_data" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PORT=8000

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        gosu \
        libsndfile1 \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /srv/requirements.txt

# Pascal has no bf16 and its fp16 rate is 1:64, so everything here runs fp32. There is
# nothing to gain from a half-precision build on this card.
RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        "torch==${TORCH_VERSION}+${TORCH_CUDA}" \
        "torchaudio==${TORCH_VERSION}+${TORCH_CUDA}" \
        --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    && python -m pip install -r /srv/requirements.txt

# Bake the NLTK corpora in. g2p_en downloads these on first use otherwise, which turns
# the first request into a network call -- and fails outright on an air-gapped host.
# Both tagger names are fetched because nltk 3.9 renamed it and g2p_en asks for the old.
RUN python -m nltk.downloader -d "${NLTK_DATA}" \
        averaged_perceptron_tagger \
        averaged_perceptron_tagger_eng \
        cmudict

COPY app /srv/app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /config

ARG VERSION=dev
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

ENV APP_VERSION=${VERSION}

LABEL org.opencontainers.image.title="WavLM Phoneme Validator" \
      org.opencontainers.image.description="Phoneme-level pronunciation scoring API with explicit expected-pronunciation input" \
      org.opencontainers.image.source="https://github.com/KenGrinder/wavlm-phoneme-validator" \
      org.opencontainers.image.url="https://github.com/KenGrinder/wavlm-phoneme-validator" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

VOLUME ["/config"]
EXPOSE 8000

# Long start period on purpose: the first boot pulls a 1.2 GB checkpoint plus the WavLM
# and XLSR backbones into /config. The check hits /health (liveness) rather than /ready
# (models loaded), or Docker would kill the container part-way through that download.
HEALTHCHECK --interval=30s --timeout=10s --start-period=25m --retries=3 \
    CMD ["python", "-m", "app.healthcheck"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips=${FORWARDED_ALLOW_IPS:-127.0.0.1}"]
