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
from auth import active_tokens, COOKIE_NAME, SECRET_KEY, ALGORITHM
from ai.background import enqueue_research
from ai.chat_handler import process_chat_message
from ai.vector_store import ingest_pdf_into_vector_store, ingest_url_into_vector_store

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PUBSUB_CHANNEL = "research_done"


# --- Socket.io setup ---

sio = socketio.AsyncServer( 
    async_mode="asgi",
    cors_allowed_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
        "https://lexis-pi.vercel.app/"
        "https://lexis-pi.vercel.app/login"
    ],
) 

# Maps socket sid → user_id for the duration of a connection
socket_sessions: dict[str, int] = {}


# --- FastAPI setup ---

async def _redis_pubsub_listener(sio_instance):
    """
    Subscribes to the Redis "research_done" channel.
    When the RQ worker publishes a completed alert, this relays
    the event to all connected socket.io clients.
    """
    r = aioredis.from_url(REDIS_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe(PUBSUB_CHANNEL)
    print("[pubsub] Listening for research_done events")

    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            alert_data = json.loads(message["data"])
            await sio_instance.emit("new_alert", {"alert": alert_data})
            print(f"[pubsub] Forwarded alert for case {alert_data.get('case_id')}")
        except Exception as error:
            print(f"[pubsub] Failed to relay message: {error}")


def _reenqueue_stuck_cases():
    """
    On server startup, check Postgres for cases that were left in "pending"
    or "processing" state (e.g. worker crashed or laptop was closed).
    Re-enqueue only those not already sitting in the Redis queue, so we
    never create duplicate jobs for the same case.
    """
    import redis as redis_sync
    from rq import Queue as RQQueue
    from rq.job import Job

    conn = redis_sync.Redis.from_url(REDIS_URL)
    q = RQQueue("legal", connection=conn)

    # Collect case_ids that already have a queued job
    already_queued: set[int] = set()
    for job_id in q.job_ids:
        try:
            job = Job.fetch(job_id, connection=conn)
            if job.args:
                already_queued.add(job.args[0])
        except Exception:
            pass

    with Session(engine) as session:
        stuck = session.exec(
            select(Case).where(Case.status.in_(["pending", "processing"]))
        ).all()

    for case in stuck:
        if case.id in already_queued:
            print(f"[startup] Case {case.id} already in queue — skipping")
        else:
            print(f"[startup] Re-enqueuing stuck case {case.id} (was '{case.status}')")
            enqueue_research(case.id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    os.makedirs("uploads", exist_ok=True)

    # Re-enqueue any research jobs that were interrupted before shutdown
    _reenqueue_stuck_cases()

    # Start the pub/sub relay in a background task (runs for the app lifetime)
    listener_task = asyncio.create_task(_redis_pubsub_listener(sio))

    yield

    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass


fastapi_app = FastAPI(
    lifespan=lifespan,
    swagger_ui_parameters={"withCredentials": True},
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
        "https://lexis-pi.vercel.app/"
        "https://lexis-pi.vercel.app/login"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastapi_app.include_router(auth_router.router)
fastapi_app.include_router(upload.router)
fastapi_app.include_router(cases.router)
fastapi_app.include_router(files.router)
fastapi_app.include_router(alerts.router)
fastapi_app.include_router(messages.router)

@fastapi_app.get("/")
def root():
    return {"message": "Legal Assistant Server is running"}

# --- Socket.io event handlers ---

@sio.event
async def connect(sid, environ, auth):
    # Read the HttpOnly access_token cookie from the socket handshake
    cookie_header = environ.get("HTTP_COOKIE", "")
    cookie = SimpleCookie()
    cookie.load(cookie_header)
    token_morsel = cookie.get(COOKIE_NAME)
    token = token_morsel.value if token_morsel else ""

    if not token:
        print(f"[socket] Rejected — no token: {sid}")
        return False

    # Fast path: token is already in the in-memory session store
    if token in active_tokens:
        socket_sessions[sid] = active_tokens[token]
        print(f"[socket] Client connected: {sid}, user_id: {active_tokens[token]}")
        return

    # Slow path: server may have restarted and lost active_tokens.
    # Decode the JWT directly so valid sessions survive restarts.
    try:
        from jose import jwt as jose_jwt, JWTError
        payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
        active_tokens[token] = user_id   # restore so HTTP routes also work
        socket_sessions[sid] = user_id
        print(f"[socket] Client reconnected after restart: {sid}, user_id: {user_id}")
    except Exception:
        print(f"[socket] Rejected invalid/expired token: {sid}")
        return False


@sio.event
async def disconnect(sid):
    user_id = socket_sessions.pop(sid, None)
    print(f"[socket] Client disconnected: {sid}, user_id: {user_id}")


@sio.event
async def start_case(sid, data):
    """
    Fired by the client when the user clicks 'Start Case Strategy'.
    Saves the case to the DB and kicks off background research.
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

    print(f"[socket] Case {case_id} created by {sid} (user_id={user_id})")

    await sio.emit("case_created", {"case_id": case_id}, to=sid)

    # Ingest all PDFs, URLs, and Images into the vector store so they are searchable.
    from ai.vector_store import ingest_pdf_into_vector_store, ingest_url_into_vector_store, ingest_image_into_vector_store
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

    # Enqueue background research — persists across server/worker restarts
    enqueue_research(case_id)


@sio.event
async def chat_message(sid, data):
    """
    Fired when the user sends a message in the case intelligence stream.

    Expected data:
    {
        "case_id": 1,
        "content": "What are the key obligations in the contract?"
    }
    """
    case_id = data.get("case_id")
    content = data.get("content", "").strip()

    if not case_id or not content:
        return

    print(f"[socket] chat_message for case {case_id} from {sid}: {content[:60]}")

    asyncio.create_task(process_chat_message(case_id, content, sio, sid))


# --- Combine FastAPI + Socket.io into one ASGI app ---

app = socketio.ASGIApp(sio, fastapi_app)
