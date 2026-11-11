import sys
import os
import types
from dotenv import load_dotenv

# LangChain v0.2/v0.3 compat monkeypatch for older Langfuse SDK
try:
    import langchain_core.callbacks
    import langchain_core.agents
    import langchain_core.documents

    callbacks_mod = types.ModuleType("langchain.callbacks")
    callbacks_base_mod = types.ModuleType("langchain.callbacks.base")
    callbacks_base_mod.BaseCallbackHandler = langchain_core.callbacks.BaseCallbackHandler
    sys.modules["langchain.callbacks"] = callbacks_mod
    sys.modules["langchain.callbacks.base"] = callbacks_base_mod

    schema_mod = types.ModuleType("langchain.schema")
    schema_agent_mod = types.ModuleType("langchain.schema.agent")
    schema_agent_mod.AgentAction = langchain_core.agents.AgentAction
    schema_agent_mod.AgentFinish = langchain_core.agents.AgentFinish

    schema_document_mod = types.ModuleType("langchain.schema.document")
    schema_document_mod.Document = langchain_core.documents.Document

    sys.modules["langchain.schema"] = schema_mod
    sys.modules["langchain.schema.agent"] = schema_agent_mod
    sys.modules["langchain.schema.document"] = schema_document_mod
except ImportError:
    pass

from langfuse.callback import CallbackHandler

load_dotenv()

def get_langfuse_handler(
    session_id: str | None = None,
    user_id: str | None = None,
    trace_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    """
    Initializes and returns the Langfuse CallbackHandler if API keys are configured.
    Attaches optional session_id, user_id, trace_name, and tags to populate
    the Langfuse Sessions and Users dashboards.
    Returns None otherwise.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    
    if public_key and secret_key:
        return CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            session_id=str(session_id) if session_id is not None else None,
            user_id=str(user_id) if user_id is not None else None,
            trace_name=trace_name,
            tags=tags,
            metadata=metadata,
        )
    return None

