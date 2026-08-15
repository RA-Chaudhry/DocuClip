#!/bin/bash
set -e

echo "=================================================="
echo "   Starting DocuClip Unified AI Pipeline Server   "
echo "=================================================="

# Ensure working directories exist with proper permissions
mkdir -p /app/clips /app/uploads /tmp/shared
chmod -R 777 /app/clips /app/uploads /tmp/shared 2>/dev/null || true

# 1. Start local Redis daemon if no external REDIS_URL is provided or pointing to localhost
if [ -z "$REDIS_URL" ] || [ "$REDIS_URL" = "redis://localhost:6379/0" ] || [ "$REDIS_URL" = "redis://127.0.0.1:6379/0" ]; then
    echo "Starting local Redis daemon on port 6379 (maxmemory 256mb, allkeys-lru)..."
    redis-server --daemonize yes --protected-mode no --maxmemory 256mb --maxmemory-policy allkeys-lru || true
    export REDIS_URL="redis://127.0.0.1:6379/0"
fi

# 2. Start Celery worker in background for video processing
echo "Starting Celery GPU/IO Worker..."
celery -A app.worker.celery_app.celery_app worker \
    --hostname=worker@%h \
    --loglevel=info \
    -Q io_queue,gpu_queue \
    --pool=prefork \
    --concurrency=${CELERY_CONCURRENCY:-1} \
    --max-memory-per-child=300000 \
    --max-tasks-per-child=25 &

# 3. Determine port (Hugging Face Spaces uses 7860, standard Docker uses 8000)
PORT_TO_USE=${PORT:-7860}
echo "Starting FastAPI Server on 0.0.0.0:${PORT_TO_USE}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT_TO_USE" --workers 1
