"""
background.py

Thin helper: enqueue a research job onto the RQ "legal" queue.
The actual job logic lives in worker.py and runs in the RQ worker process.
"""

import redis
import os
from rq import Queue

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


def get_queue() -> Queue:
    conn = redis.from_url(REDIS_URL)
    return Queue("legal", connection=conn)


def enqueue_research(case_id: int) -> None:
    """Push a research job for case_id onto the Redis queue."""
    queue = get_queue()
    # job_timeout: max seconds the worker process may run before RQ kills it.
    # deepseek-r1:8b research can take 10-20 min; 1800s gives a safe margin.
    # Without this, RQ's default 180s timeout fires mid-LLM-call → SIGABRT.
    queue.enqueue("worker.research_job", case_id, job_timeout=1800)
    print(f"[background] Enqueued research job for case {case_id}")
