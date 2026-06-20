# Legal Assistant — Backend Server

> The intelligence engine powering the Legal Assistant. Built with FastAPI, Socket.IO, LangGraph, and CrewAI. Handles case management, real-time chat, voice calls, vector ingestion, and background legal research.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
  - [HTTP API (FastAPI)](#http-api-fastapi)
  - [Real-Time Duplex (Socket.IO)](#real-time-duplex-socketio)
  - [Background Workers (RQ + Redis)](#background-workers-rq--redis)
- [AI Pipelines](#ai-pipelines)
  - [Intelligence Chat Pipeline (LangGraph)](#intelligence-chat-pipeline-langgraph)
  - [Background Research Pipeline (CrewAI)](#background-research-pipeline-crewai)
  - [Voice Call Pipeline](#voice-call-pipeline)
  - [Vector Ingestion Pipeline](#vector-ingestion-pipeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the Services](#running-the-services)
- [Database Schema (SQLModel)](#database-schema-sqlmodel)
- [Route Handlers](#route-handlers)
- [AI Modules](#ai-modules)
- [Deployment](#deployment)

---

## Overview

The backend orchestrates all data management and AI reasoning for the Legal Assistant. Key capabilities include:

1. **Authentication & Case Management:** Secure JWT-based auth (via HTTP-only cookies), case creation, and evidence vault management.
2. **Real-time AI Chat:** A LangGraph pipeline that reads vault documents (RAG via Chroma) and performs adversarial web searches (Tavily) to synthesize strategic legal advice.
3. **Voice AI Agent:** Full-duplex voice interactions capturing mic audio, transcribing it, generating LLM responses, and streaming back TTS audio.
4. **Background Research:** An asynchronous CrewAI worker that scours the web for litigation precedents, regulatory fines, and class actions related to new cases, raising alerts in the UI.
5. **Multimodal Evidence:** Cloudinary integration for storing PDFs and images. Vision models extract text from images, while PyPDF extracts text from documents for vector ingestion.

---

## Tech Stack

| Component            | Technology                                                                                   |
| -------------------- | -------------------------------------------------------------------------------------------- |
| **Framework**        | [FastAPI](https://fastapi.tiangolo.com/) (REST) + [Socket.IO](https://python-socketio.readthedocs.io/) (Real-time) |
| **Database**         | [Neon (PostgreSQL)](https://neon.tech/) via [SQLModel](https://sqlmodel.tiangolo.com/) / SQLAlchemy |
| **Vector Store**     | [ChromaDB](https://www.trychroma.com/) (Local persistence)                                   |
| **Background Jobs**  | [RQ (Redis Queue)](https://python-rq.org/) + Redis PubSub                                   |
| **File Storage**     | [Cloudinary](https://cloudinary.com/)                                                        |
| **LLM Orchestration**| [LangGraph](https://langchain-ai.github.io/langgraph/) (Chat) + [CrewAI](https://www.crewai.com/) (Background) |
| **LLMs (via API)**   | [Mistral](https://mistral.ai/) (`mistral-small-latest`, `mistral-large-latest`, `mistral-embed`) |
| **Voice (STT/TTS)**  | Mistral / Voxtral (`voxtral-mini-latest`, `voxtral-mini-tts-2603`)                           |
| **Web Search**       | [Tavily](https://tavily.com/)                                                                |
| **Auth**             | JWT (python-jose) + bcrypt                                                                   |

---

## Architecture

### HTTP API (FastAPI)

Handles all synchronous REST requests: login/logout, fetching case lists, uploading files, and retrieving historical messages or alerts. Runs via `uvicorn` using the `main:app` ASGI wrapper.

### Real-Time Duplex (Socket.IO)

Mounted alongside FastAPI. Handles:
- **Chat:** Client sends `chat_message`. Server runs LangGraph in a thread executor, emitting `stage_update` (e.g., "Auditing documents...") and finally `ai_response`.
- **Voice:** Client streams `audio_chunk`. Server buffers, transcribes, queries LLM, and emits `call_ai_text` + `call_ai_audio`. Handles interruption signals.
- **Alerts:** Server listens to a Redis `research_done` PubSub channel and broadcasts `new_alert` to connected clients.

### Background Workers (RQ + Redis)

Deep legal research can take 5-15 minutes.
- The FastAPI server enqueues jobs to Redis.
- An RQ worker process (`run_worker.py`) picks up the job, runs a CrewAI pipeline, saves the `Alert` to Postgres, and publishes a `research_done` message to Redis.
- A background task in `main.py` listens to Redis and relays the alert via Socket.IO.

---

## AI Pipelines

### Intelligence Chat Pipeline (LangGraph)

Defined in `ai/graph.py` and `ai/chat_handler.py`.

A 3-node graph for answering user questions:
1. **Analyst Node (Parallel):** Uses Chroma vector search to find relevant chunks in the vault. Extracts direct quotes, contradictions, and admissions. Strict anti-hallucination rules.
2. **Researcher Node (Parallel):** Uses Tavily to build adversarial web queries based on the case context to find real-world lawsuits and precedents.
3. **Strategist Node:** Synthesizes the Analyst and Researcher outputs into a rigid IRAC (Issue, Rule, Analysis, Conclusion) format with 3 Next Moves.

### Background Research Pipeline (CrewAI)

Defined in `ai/crew.py` and `worker.py`.

Triggered when a new case is created or when requested by the Strategist LLM.
- **Agent:** Legal Research Hunter.
- **Task:** Dig deep into the web for regulatory enforcement actions, class actions, and product liability cases related to the opponent.
- **Output Validation:** Forces a structured Pydantic `ResearchOutput` (Liability Summary, Evidence Log, Source Index) to ensure proper citation formatting before saving as an `Alert`.

### Voice Call Pipeline

Defined in `ai/call_handler.py`.

1. **VAD (Voice Activity Detection):** Calculates RMS energy of incoming PCM16 chunks to dynamically determine speech threshold vs background noise.
2. **STT:** Sends audio window to Mistral's transcription endpoint.
3. **Chat:** Passes transcript + case context + history to `mistral-large-latest` with a specific voice persona prompt (short, conversational).
4. **TTS:** Sends AI text to Mistral's TTS endpoint.
5. **Playback:** Emits base64 audio to the client. Stops immediately if client emits `call_user_speaking`.

### Vector Ingestion Pipeline

Defined in `ai/vector_store.py` and `ai/ingestion.py`.

Each case gets an isolated Chroma collection (`case_{id}`).
- **PDFs:** `pypdf` extracts text.
- **URLs:** `BeautifulSoup` extracts text.
- **Images:** Mistral Vision model generates a descriptive summary.
- Text is chunked (800 chars, 150 overlap) and embedded via `mistral-embed`. Runs in a `BackgroundTasks` thread during the file upload HTTP request.

---

## Project Structure

```
server/
├── main.py                # ASGI entry point (FastAPI + SocketIO app)
├── database.py            # Postgres connection and table creation
├── models.py              # SQLModel schema definitions
├── auth.py                # JWT creation, validation, and decorators
├── worker.py              # RQ worker job definition (research_job)
├── run_worker.py          # Script to run the RQ worker (SimpleWorker)
├── cloudinary_client.py   # Cloudinary upload helpers
├── .env                   # Environment variables
├── pyproject.toml         # Python project metadata and dependencies
├── uv.lock                # Locked dependency tree for 'uv'
├── Dockerfile             # Container definition
├── .github/workflows/     # GitHub Actions CI/CD workflows
│
├── routes/                # FastAPI HTTP route handlers
│   ├── auth.py            # Login, registration, admin setup
│   ├── cases.py           # Case listing, URL summarization, context updating
│   ├── upload.py          # Cloudinary upload + background ingestion trigger
│   ├── alerts.py          # Alert listing, marking read, archiving
│   ├── messages.py        # Chat history retrieval
│   └── files.py           # Fallback local file serving
│
└── ai/                    # Core Intelligence Engine
    ├── background.py      # RQ enqueue wrapper
    ├── call_handler.py    # Voice session state machine, VAD, STT/TTS calls
    ├── chat_handler.py    # SocketIO chat message handler (invokes LangGraph)
    ├── chat_llm.py        # LangGraph Node 3: Strategist
    ├── analyst.py         # LangGraph Node 1a: Analyst (Vector Store RAG)
    ├── researcher.py      # LangGraph Node 1b: Researcher (Tavily Search)
    ├── graph.py           # LangGraph pipeline compilation and execution
    ├── state.py           # TypedDict state definition for LangGraph
    ├── crew.py            # CrewAI configuration and prompt for background research
    ├── model_providers.py # Centralized Langchain LLM/Embedding instantiation
    ├── vector_store.py    # Chroma DB initialization, chunking, and insertion
    ├── ingestion.py       # PDF/URL text extraction utilities
    ├── image_handler.py   # Vision LLM image description
    └── summarizer.py      # Markdown generator for URL evidence
```

---

## Getting Started

### Prerequisites

- **Python** 3.12+
- **uv** (recommended) or pip
- **Redis** server running locally (`redis-server`)
- **PostgreSQL** database (Neon recommended)

### Installation

```bash
cd server
uv sync
# OR python -m venv .venv && source .venv/bin/activate && pip install -e .
```

### Environment Variables

Create a `.env` file in the `server/` directory:

```env
# Database
DATABASE_URL=postgresql://user:pass@ep-restless...neon.tech/neondb

# Redis (for background jobs and PubSub)
REDIS_URL=redis://localhost:6379

# Auth
JWT_SECRET_KEY=your_super_secret_key

# AI Providers
MISTRAL_API_KEY=your_mistral_key
TAVILY_API_KEY=your_tavily_key

# Storage
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Optional: Fixes fork safety issues on macOS for RQ
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

### Running the Services

You need three terminal tabs running simultaneously:

**1. FastAPI + Socket.IO Server:**
```bash
uv run uvicorn main:app --reload
```

**2. Background Research Worker:**
```bash
# macOS users: Ensure OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES is exported
uv run python run_worker.py
```

**3. Redis Server:**
```bash
redis-server
```

*(Optional)* Run the RQ dashboard to monitor jobs:
```bash
uv run rq-dashboard --redis-url redis://localhost:6379
```

---

## Database Schema (SQLModel)

Defined in `models.py`.

- **`User`**: Authentication credentials, roles (`admin` or `employee`), and activation status.
- **`Case`**: Represents a litigation project. Contains `context`, `urls_json`, `pdf_paths_json`, `image_paths_json`, and a `status` tracking background research.
- **`CaseMessage`**: Persists chat history. Links to `Case`. Stores role (`user` or `ai`), text content, and structured `citation_json`.
- **`Alert`**: Research findings generated by the background worker. Links to `Case`. Contains markdown `summary`, `ai_reasoning`, severity, and read/archived status.

---

## Route Handlers

- **`auth.py`**: Multi-step login flow (`/check-email` -> `/login` or `/set-password`), token validation, and admin user management.
- **`cases.py`**: Fetches formatted case lists for the UI, triggers URL summarization, and allows appending context to an existing case (which re-triggers research).
- **`upload.py`**: Receives multipart form data, uploads to Cloudinary, stores the URL in Postgres, and fires `BackgroundTasks` to ingest into Chroma.
- **`alerts.py`**: Fetches unread/read alerts and handles bulk archiving.
- **`messages.py`**: Loads historical chat logs for a given case modal.

---

## AI Modules

The `ai/` folder contains all intelligence logic:

- **`model_providers.py`**: Single source of truth for instantiating Chat (`get_chat_model`) and Embedding (`get_embeddings`) models. Currently wired to Mistral APIs.
- **`vector_store.py`**: Handles ChromaDB operations. Each case uses an isolated collection (`case_{id}`). Includes specific ingestion functions for URLs, PDFs, and Images (via Vision descriptions).
- **`graph.py`**: Defines the LangGraph StateGraph. Executes Analyst and Researcher in parallel, fanning in to the Strategist.
- **`crew.py`**: Configures the CrewAI agent for deep, multi-step background research. Includes a custom patch to fix Mistral API message sequence constraints.
- **`call_handler.py`**: The `CallSession` class manages the state machine for voice. Handles RMS-based dynamic noise thresholding to detect when a user starts/stops speaking.

---

## Deployment

The server is deployed to a Hostinger VPS using Docker Compose.

**Key deployment notes:**
- The deployment is automated via GitHub Actions (configured in `.github/workflows/deploy.yml`).
- Docker Compose manages four services: `api`, `worker`, `redis`, `dashboard` (rq-dashboard), and `caddy` (reverse proxy).
- Ensure all environment variables are set in the `.env` file on the VPS server under `/root/legal-assistant/server/.env`.
- Caddy automatically provisions SSL certificates for `lexis-api.discoverbenix.com`.
