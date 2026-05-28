"""
run_worker.py

Start the RQ worker using SimpleWorker, which runs jobs in the same process
without forking. This avoids SIGABRT crashes caused by native C libraries
(LiteLLM, httpx) that don't survive a fork() call.

Usage:
    cd server
    source .venv/bin/activate
    python run_worker.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

import redis
from rq import Queue 
from rq.worker import SimpleWorker

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

conn = redis.from_url(REDIS_URL)
q = Queue("legal", connection=conn)

print(f"[worker] Starting SimpleWorker on queue 'legal' ({REDIS_URL})")
worker = SimpleWorker([q], connection=conn)
worker.work(with_scheduler=False) 


# OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES rq worker --url redis://localhost:6379 legal

# rq-dashboard --redis-url redis://localhost:6379
 