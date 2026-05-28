"""
main.py

Main ASGI server application combining FastAPI (HTTP REST endpoints) and Socket.IO (Real-time duplex communication).
Includes full support for Case Intelligence Chat and Real-time Voice Call pipelines.
"""

import json
import os
import asyncio
from contextlib import asynccontextmanager
from http.cookies import SimpleCookie

import redis.asyncio as aioredis
import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select 

from database import create_tables, engine
from models import Case
from routes import upload, cases, files, alerts, messages 
from routes import auth as auth_router
from auth import revoked_tokens, COOKIE_NAME, SECRET_KEY, ALGORITHM
from ai.background import enqueue_research
from ai.chat_handler import process_chat_message
from ai.vector_store import ingest_pdf_into_vector_store, ingest_url_into_vector_store
from ai.call_handler import CallSession, active_call_sessions

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PUBSUB_CHANNEL = "research_done"


# --- Socket.IO setup ---

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
        # Production origin (no trailing slash)
        "https://lexis-pi.vercel.app",
    ],
)

# Maps socket sid -> user_id for the duration of the connection
socket_sessions: dict[str, int] = {}


# --- FastAPI setup & Redis PubSub Relayer ---

async def _redis_pubsub_listener(sio_instance):
    """
    Subscribes to the Redis 'research_done' channel.
    Relays research alerts to connected socket clients.
    """
    r = aioredis.from_url(REDIS_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe(PUBSUB_CHANNEL)
    print("[pubsub] Listening for research_done events")

    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            payload = json.loads(message["data"])
            if payload.get("type") == "research_error":
                await sio_instance.emit("research_error", payload)
                print(f"[pubsub] Forwarded error for case {payload.get('case_name')}")
            else:
                await sio_instance.emit("new_alert", {"alert": payload})
                print(f"[pubsub] Forwarded alert for case {payload.get('case_id')}")
        except Exception as error:
            print(f"[pubsub] Failed to relay message: {error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles server startup and shutdown lifecycles.
    - Creates DB tables
    - Spawns Redis PubSub listener task
    - Re-enqueues pending/processing research jobs that crashed
    """
    # Create DB tables if they don't exist
    create_tables()

    # Start the pubsub listener in the background
    pubsub_task = asyncio.create_task(_redis_pubsub_listener(sio))

    # Re-enqueue crashed or unfinished background research jobs
    import redis as redis_sync
    from rq import Queue as RQQueue
    from rq.job import Job

    try:
        conn = redis_sync.Redis.from_url(REDIS_URL)
        q = RQQueue("legal", connection=conn)

        # Gather case IDs that are already queued
        already_queued: set[int] = set()
        for job_id in q.job_ids:
            try:
                job = Job.fetch(job_id, connection=conn)
                if job and job.args:
                    already_queued.add(int(job.args[0]))
            except Exception:
                continue

        # Scan Postgres for pending or processing cases
        with Session(engine) as session:
            statement = select(Case).where(Case.status.in_(["pending", "processing"]))
            crashed_cases = session.exec(statement).all()
            
            for case in crashed_cases:
                if case.id not in already_queued:
                    print(f"[lifespan] Re-enqueuing crashed case research: {case.id}")
                    enqueue_research(case.id)
    except Exception as error:
        print(f"[lifespan] Startup re-enqueue error: {error}")

    yield

    # Clean up background pubsub task
    pubsub_task.cancel()
    try:
        await pubsub_task
    except asyncio.CancelledError:
        pass


fastapi_app = FastAPI(lifespan=lifespan)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
        "https://lexis-pi.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
fastapi_app.include_router(auth_router.router)
fastapi_app.include_router(upload.router)
fastapi_app.include_router(cases.router)
fastapi_app.include_router(files.router)
fastapi_app.include_router(alerts.router)
fastapi_app.include_router(messages.router)


@fastapi_app.get("/")
def root():
    return {"message": "Legal Assistant Server is running"}


# --- Socket.IO event handlers ---

@sio.event
async def connect(sid, environ, auth):
    """
    Authenticates Socket.IO connections.
    Reads access token cookie from HTTP handshake headers.
    """
    cookie_header = environ.get("HTTP_COOKIE", "")
    cookie = SimpleCookie()
    cookie.load(cookie_header)
    token_morsel = cookie.get(COOKIE_NAME)
    token = token_morsel.value if token_morsel else ""

    if not token:
        print(f"[socket] Rejected — no token: {sid}")
        return False

    # Check if token is revoked
    if token in revoked_tokens:
        print(f"[socket] Rejected — token is revoked: {sid}")
        return False

    # Decode and restore JWT directly
    try:
        from jose import jwt as jose_jwt
        payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
        socket_sessions[sid] = user_id
        print(f"[socket] Client connected: {sid}, user_id: {user_id}")
        return True
    except Exception:
        print(f"[socket] Rejected invalid/expired token: {sid}")
        return False


@sio.event
async def disconnect(sid):
    """Cleans up session records when socket disconnects."""
    user_id = socket_sessions.pop(sid, None)
    
    # End and clean up any voice call sessions
    session = active_call_sessions.pop(sid, None)
    if session:
        await session.stop()

    print(f"[socket] Client disconnected: {sid}, user_id: {user_id}")


@sio.event
async def start_case(sid, data):
    """
    Fired when a user starts analyzing a new case.
    Ingests files and enqueues research worker jobs.
    """
    context = data.get("context", "")
    urls = data.get("urls", [])
    pdf_paths = data.get("pdf_paths", [])
    image_paths = data.get("image_paths", [])
    user_id = socket_sessions.get(sid)

    new_case = Case(
        context=context,
        urls_json=json.dumps(urls),
        pdf_paths_json=json.dumps(pdf_paths),
        image_paths_json=json.dumps(image_paths),
        status="pending",
        user_id=user_id,
    )

    with Session(engine) as session:
        session.add(new_case)
        session.commit()
        session.refresh(new_case)
        case_id = new_case.id

    # Ingest PDFs, URLs, and images in a background thread executor
    from ai.vector_store import (
        ingest_pdf_into_vector_store,
        ingest_url_into_vector_store,
        ingest_image_into_vector_store
    )
    loop = asyncio.get_event_loop()

    for pdf_path in pdf_paths:
        await loop.run_in_executor(
            None,
            ingest_pdf_into_vector_store,
            case_id,
            pdf_path,
        )

    for url in urls:
        await loop.run_in_executor(
            None,
            ingest_url_into_vector_store,
            case_id,
            url,
        )

    for img_path in image_paths:
        await loop.run_in_executor(
            None,
            ingest_image_into_vector_store,
            case_id,
            img_path,
        )

    # Enqueue background research job
    enqueue_research(case_id)


@sio.event
async def chat_message(sid, data):
    """Fired when user sends a chat message in the Intelligence Stream."""
    case_id = data.get("case_id")
    content = data.get("content", "").strip()

    if not case_id or not content:
        return

    print(f"[socket] chat_message for case {case_id} from {sid}: {content[:60]}")
    asyncio.create_task(process_chat_message(case_id, content, sio, sid))


# --- Voice Call event handlers ---

@sio.event
async def start_call_session(sid, data):
    """Fired when user clicks 'Call Assistant'."""
    case_id = data.get("case_id")
    user_id = socket_sessions.get(sid)

    if not case_id:
        await sio.emit("call_error", {
            "message": "No case_id provided.",
        }, to=sid)
        return

    # Clean up any active sessions for this client first
    if sid in active_call_sessions:
        old_session = active_call_sessions[sid]
        await old_session.stop()
        del active_call_sessions[sid]

    session = CallSession(sio, sid, int(case_id))
    active_call_sessions[sid] = session
    await session.start()

    print(f"[socket] Call session started for case {case_id} by {sid} (user_id={user_id})")
    await sio.emit("call_session_started", {
        "case_id": case_id,
    }, to=sid)


@sio.event
async def audio_chunk(sid, data):
    """Receives binary PCM16 audio chunks from browser microphone."""
    session = active_call_sessions.get(sid)
    if not session:
        return

    if isinstance(data, (bytes, bytearray)):
        await session.handle_audio_chunk(bytes(data))


@sio.event
async def call_user_speaking(sid):
    """Browser signals the user has started speaking (interruption event)."""
    session = active_call_sessions.get(sid)
    if session:
        await session.handle_interruption()


@sio.event
async def call_ai_speaking_done(sid):
    """Browser signals that the AI response audio has completed playback."""
    session = active_call_sessions.get(sid)
    if session:
        session.mark_ai_done_speaking()


@sio.event
async def end_call_session(sid):
    """Cleans up the call session when requested."""
    session = active_call_sessions.pop(sid, None)
    if session:
        await session.stop()
        print(f"[socket] Call session ended for case {session.case_id} by {sid}")


# --- Wrap combined app ---
app = socketio.ASGIApp(sio, fastapi_app)