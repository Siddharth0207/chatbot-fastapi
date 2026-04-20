"""FastAPI backend for diamond chat with fast-path inference and resilient fallbacks."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from main import search_diamonds_db
from utils.config import get_settings
from utils.logger import logging
from utils.prompts import agent_prompt


TOOL_QUERY_HINTS = {
    "diamond",
    "carat",
    "clarity",
    "color",
    "cut",
    "polish",
    "symmetry",
    "fluorescence",
    "lab",
    "gia",
    "igi",
    "vvs",
    "vs1",
    "vs2",
    "si1",
    "si2",
    "shape",
    "round",
    "oval",
    "pear",
    "emerald",
    "radiant",
    "princess",
    "recommend",
    "find",
    "search",
    "inventory",
    "budget",
    "price",
}

CHAT_SYSTEM_PROMPT = (
    "You are a friendly, concise diamond concierge assistant. "
    "Hold natural conversation, answer jewelry and diamond education questions, "
    "and keep responses brief unless the user asks for detail."
)


class DiamondQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User message to the assistant.")
    session_id: str | None = Field(
        default=None,
        description="Optional session id. If omitted, x-session-id header is used or a new session is created.",
    )


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, description="Optional display title for the chat session.")


@dataclass
class ChatSession:
    session_id: str
    title: str = "New chat"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[dict[str, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


app = FastAPI()
settings = get_settings()
session_store: dict[str, ChatSession] = {}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_title(query: str, max_len: int = 72) -> str:
    clean = " ".join(query.split())
    if len(clean) <= max_len:
        return clean or "New chat"
    return f"{clean[:max_len - 1]}..."


def _coerce_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if text_value:
                    parts.append(str(text_value))
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()

    return str(content).strip()


def _requires_tool_lookup(query: str) -> bool:
    lower_query = query.lower()
    if any(token in lower_query for token in TOOL_QUERY_HINTS):
        return True
    return "diamond" in lower_query and any(ch.isdigit() for ch in lower_query)


def _history_as_messages(
    session: ChatSession,
    max_messages: int,
) -> list[HumanMessage | AIMessage]:
    history: list[HumanMessage | AIMessage] = []
    for item in session.messages[-max_messages:]:
        content = (item.get("content") or "").strip()
        if not content:
            continue

        role = item.get("role")
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    return history


def _serialize_session(session: ChatSession, include_messages: bool = False) -> dict:
    payload: dict[str, object] = {
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "message_count": len(session.messages),
    }
    if include_messages:
        payload["messages"] = session.messages
    return payload


def _create_session(session_id: str | None = None, title: str | None = None) -> ChatSession:
    resolved_id = session_id or str(uuid.uuid4())
    resolved_title = title.strip() if title and title.strip() else "New chat"
    session = ChatSession(session_id=resolved_id, title=resolved_title)
    session_store[resolved_id] = session
    return session


def _get_or_create_session(session_id: str | None = None) -> ChatSession:
    if session_id and session_id in session_store:
        return session_store[session_id]
    return _create_session(session_id=session_id)


cors_origins = _parse_csv(settings.CORS_ALLOW_ORIGINS)
cors_methods = _parse_csv(settings.CORS_ALLOW_METHODS)
cors_headers = _parse_csv(settings.CORS_ALLOW_HEADERS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=cors_methods,
    allow_headers=cors_headers,
)

tools = [search_diamonds_db]
chat_llm: ChatNVIDIA | None = None
tool_agent = None
chat_model_init_lock = asyncio.Lock()
tool_agent_init_lock = asyncio.Lock()


def _create_chat_llm() -> ChatNVIDIA:
    return ChatNVIDIA(
        model=settings.NVIDIA_CHAT_MODEL,
        task="chat",
        temperature=settings.NVIDIA_TEMPERATURE,
        top_p=settings.NVIDIA_TOP_P,
        max_tokens=settings.NVIDIA_CHAT_MAX_TOKENS,
        stream=False,
        api_key=settings.NVIDIA_API_KEY,
    )


def _create_tool_agent():
    tool_llm = ChatNVIDIA(
        model=settings.NVIDIA_TOOL_MODEL,
        task="chat",
        temperature=settings.NVIDIA_TEMPERATURE,
        top_p=settings.NVIDIA_TOP_P,
        max_tokens=settings.NVIDIA_TOOL_MAX_TOKENS,
        stream=False,
        api_key=settings.NVIDIA_API_KEY,
    )
    return create_tool_calling_agent(tool_llm, tools, agent_prompt)


async def _get_chat_model() -> ChatNVIDIA:
    global chat_llm
    if chat_llm is not None:
        return chat_llm

    async with chat_model_init_lock:
        if chat_llm is None:
            chat_llm = _create_chat_llm()
    return chat_llm


async def _get_tool_agent():
    global tool_agent
    if tool_agent is not None:
        return tool_agent

    async with tool_agent_init_lock:
        if tool_agent is None:
            tool_agent = _create_tool_agent()
    return tool_agent


async def _run_chat_path(query: str, session: ChatSession) -> str:
    model = await _get_chat_model()
    history = _history_as_messages(session, settings.MEMORY_MAX_MESSAGES)
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT), *history, HumanMessage(content=query)]

    response = await asyncio.wait_for(
        model.ainvoke(messages),
        timeout=settings.CHAT_TIMEOUT_SECONDS,
    )
    summary = _coerce_text(getattr(response, "content", ""))
    return summary or "I could not generate a response for that request."


async def _run_tool_path(query: str, session: ChatSession) -> str:
    agent = await _get_tool_agent()
    history = _history_as_messages(session, settings.MEMORY_MAX_MESSAGES)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.AGENT_VERBOSE,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
        max_execution_time=float(settings.AGENT_TIMEOUT_SECONDS),
        handle_parsing_errors=True,
    )

    result = await asyncio.wait_for(
        agent_executor.ainvoke({"input": query, "chat_history": history}),
        timeout=settings.AGENT_TIMEOUT_SECONDS,
    )
    summary = _coerce_text(result.get("output", ""))
    return summary or "I could not generate a response for that request."


@app.get("/")
async def read_root() -> dict:
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the Diamond Query API"}


@app.get("/sessions")
async def list_sessions() -> dict:
    sessions = sorted(session_store.values(), key=lambda item: item.updated_at, reverse=True)
    return {"status": "success", "sessions": [_serialize_session(session) for session in sessions]}


@app.post("/sessions")
async def create_session(request: CreateSessionRequest | None = None) -> dict:
    session = _create_session(title=request.title if request else None)
    return {"status": "success", "session": _serialize_session(session)}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "session": _serialize_session(session, include_messages=True)}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    removed = session_store.pop(session_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "session_id": session_id}


@app.post("/query")
async def query_diamond(request: DiamondQueryRequest, fastapi_request: Request) -> JSONResponse:
    header_session_id = fastapi_request.headers.get("x-session-id")
    resolved_session_id = request.session_id or header_session_id
    session = _get_or_create_session(resolved_session_id)

    logging.info(f"/query endpoint called. Session: {session.session_id}, Query: {request.query}")

    request_started = time.perf_counter()
    status = "success"
    mode = "chat"

    async with session.lock:
        try:
            if _requires_tool_lookup(request.query):
                mode = "tool_agent"
                summary = await _run_tool_path(request.query, session)
            else:
                mode = "chat_fast"
                summary = await _run_chat_path(request.query, session)
        except asyncio.TimeoutError:
            status = "degraded"
            if mode == "tool_agent":
                summary = (
                    "I am hitting a timeout while checking live diamond inventory. "
                    "Please retry in a few seconds."
                )
            else:
                summary = "The model service is responding slowly right now. Please retry in a few seconds."
            logging.warning(
                "Query timeout. session=%s mode=%s timeout_seconds=%s",
                session.session_id,
                mode,
                settings.AGENT_TIMEOUT_SECONDS if mode == "tool_agent" else settings.CHAT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            status = "degraded"
            if mode == "tool_agent":
                summary = (
                    "I could not complete the live inventory lookup because the upstream model returned an error. "
                    "Please retry shortly."
                )
            else:
                summary = (
                    "I ran into an upstream model error while responding. "
                    "Please retry in a few seconds."
                )
            logging.error(
                "Error in /query endpoint for session %s mode=%s: %s",
                session.session_id,
                mode,
                exc,
                exc_info=True,
            )

        session.messages.append(
            {"role": "user", "content": request.query, "created_at": _now_utc().isoformat()}
        )
        session.messages.append(
            {"role": "assistant", "content": summary, "created_at": _now_utc().isoformat()}
        )
        session.updated_at = _now_utc()
        if session.title == "New chat":
            session.title = _normalize_title(request.query)

        latency_ms = int((time.perf_counter() - request_started) * 1000)
        logging.info(
            "Query completed. session=%s status=%s mode=%s latency_ms=%s",
            session.session_id,
            status,
            mode,
            latency_ms,
        )

        response_payload = {
            "session_id": session.session_id,
            "user_query": request.query,
            "summary": summary,
            "status": status,
            "mode": mode,
            "latency_ms": latency_ms,
            "session": _serialize_session(session),
        }
        return JSONResponse(content=response_payload, headers={"x-session-id": session.session_id})
