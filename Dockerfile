# The local app (ADR-023) as a CPU-only image: the FastAPI app and the e5
# encoder on torch CPU. Generation stays outside, on the host's Ollama.
#
# No third-party text and no model weights enter this image. Only src/, ui/,
# tasks.py, requirements.txt and the synthetic demo document under evals/app/
# are copied, and .dockerignore keeps data/, runs/ and models/ out of the build
# context as well. Uploaded documents live in /data/app, a host directory
# mounted at run time; the encoder downloads into the hf-cache volume on first
# use (docker-compose.yml).

# TODO: pin this tag to a digest (python:3.14-slim@sha256:...) once the image
# can be pulled, so a rebuild cannot silently pick up a different base.
FROM python:3.14-slim

# PYTHONUNBUFFERED: the startup banner and warnings reach `docker compose logs`
# as they are printed, not when a buffer fills.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LEGALRAG_DATA_DIR=/data/app \
    HF_HOME=/home/app/.cache/huggingface

# A non-root user with a fixed uid, 1000, the usual first user on a Linux host,
# so what the app writes into the ./data/app bind mount belongs to that user.
# The directories the app writes are its own: a named volume takes the
# ownership of the directory it is mounted over.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app \
 && mkdir -p /data/app /home/app/.cache/huggingface \
 && chown -R app:app /data/app /home/app/.cache

WORKDIR /app

# torch first, from the CPU wheel index: sentence-transformers in
# requirements.txt would otherwise pull the far larger CUDA build from PyPI.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY ui/ ./ui/
COPY tasks.py ./
# The synthetic demo document (evals/app/README.md), nothing else from evals/.
COPY evals/app/ ./evals/app/

USER app

EXPOSE 8000

# GET /, the chat page. Not /api/health: that asks Ollama twice on every call.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)"]

# 0.0.0.0: inside the container, 127.0.0.1 is the container's own loopback,
# which a published port cannot reach. The app prints a WARNING that the host
# is not a loopback address; that is expected here. It stays local all the
# same: docker-compose.yml publishes the port on the host's 127.0.0.1 only,
# and the Host allowlist still holds, since a browser on the host sends
# Host: 127.0.0.1:8000.
CMD ["python", "tasks.py", "app", "--host", "0.0.0.0", "--port", "8000"]
