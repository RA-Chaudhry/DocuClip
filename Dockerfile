# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0

WORKDIR /app

# Runtime packages are installed in a separate layer so application code
# changes do not invalidate the Python dependency layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies

# Keep this COPY before COPY . . -- this is the main Docker cache boundary.
COPY requirements.txt ./requirements.txt

RUN --mount=type=cache,id=docuclip-pip,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Keep small/fast-changing additions in a separate layer. Adding or updating
# one package here will not reinstall the large requirements.txt layer.
COPY requirements.extra.txt ./requirements.extra.txt
RUN --mount=type=cache,id=docuclip-pip-extra,target=/root/.cache/pip \
    python -m pip install -r requirements.extra.txt

FROM dependencies AS development

COPY . .
RUN mkdir -p /app/clips /tmp/shared

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/app"]

FROM dependencies AS production

# Production image contains only application/runtime files and never relies
# on a source bind mount.
COPY app ./app
COPY frontend ./frontend
RUN mkdir -p /app/clips /tmp/shared

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
