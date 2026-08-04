from celery import Celery
from kombu import Queue
from app.core.config import settings

# Celery application initialize ho raha hai, jisme redis backend aur broker use kiya hai
celery_app = Celery(
    "docuclip_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.worker.tasks.io_tasks"]
)

# Dual-queue architecture definition
# io_queue text/db/network tasks ke liye hai aur gpu_queue high computational AI/ML models tasks ke liye
celery_app.conf.task_queues = (
    Queue("io_queue"),
    Queue("gpu_queue"),
)

# task_routes configuration:
# app.worker.tasks.io_ se shuru hone wale tasks io_queue me jayenge
# app.worker.tasks.gpu_ se shuru hone wale tasks gpu_queue me jayenge
celery_app.conf.task_routes = {
    "app.worker.tasks.io_*": {"queue": "io_queue"},
    "app.worker.tasks.gpu_*": {"queue": "gpu_queue"},
}

# Task modules to import explicitly when the application starts
celery_app.conf.imports = (
    "app.worker.tasks.io_tasks",
    "app.worker.tasks.gpu_tasks",
)

# Reliability settings for idempotency:
# Prefetch multiplier=1 set kiya hai taaki ek worker ek time par ek hi task fetch kare
celery_app.conf.worker_prefetch_multiplier = 1

# task_acks_late=True set kiya hai taaki failure case me task safety maintain ho sake aur task redeliver ho sake
celery_app.conf.task_acks_late = True
