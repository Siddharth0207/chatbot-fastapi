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
    """
    Pydantic model for validating incoming diamond query requests.
    
    This model is used to parse and validate JSON requests sent to the /query endpoint.
    It enforces type checking and validation at the API boundary, ensuring that only
    valid queries are processed by the application.
    
    Attributes:
        query (str): The user's natural language question or request about diamonds.
                     Must be at least 1 character long. Examples: "Find me a 2 carat 
                     diamond" or "What is clarity?"
        session_id (str | None): Unique identifier for the chat session. If not provided,
                                 the system will attempt to use the x-session-id header
                                 value or create a new session. Used for maintaining
                                 conversation context and history.
    
    Example:
        >>> request = DiamondQueryRequest(query="Show me round diamonds under $10k")
        >>> print(request.query)
        'Show me round diamonds under $10k'
    """
    query: str = Field(..., min_length=1, description="User message to the assistant.")
    session_id: str | None = Field(
        default=None,
        description="Optional session id. If omitted, x-session-id header is used or a new session is created.",
    )


class CreateSessionRequest(BaseModel):
    """
    Pydantic model for creating a new chat session.
    
    This model handles requests to the /sessions POST endpoint, allowing clients
    to optionally provide a custom title for a new conversation session.
    
    Attributes:
        title (str | None): An optional human-readable name for the chat session.
                           Examples: "Diamond Buying Guide" or "My Search"
                           If not provided or empty, defaults to "New chat" when
                           the session is created.
    
    Example:
        >>> request = CreateSessionRequest(title="My Diamond Journey")
        >>> print(request.title)
        'My Diamond Journey'
    """
    title: str | None = Field(default=None, description="Optional display title for the chat session.")


@dataclass
class ChatSession:
    """
    Represents a single conversation session between a user and the chatbot.
    
    This dataclass maintains all conversation state including the message history,
    timestamps, and metadata. It uses an async lock to ensure thread-safe operations
    when multiple concurrent requests access the same session.
    
    This is an in-memory session store. Sessions are not persisted to a database
    and will be lost when the application restarts.
    
    Attributes:
        session_id (str): Unique identifier for this chat session (UUID format).
                         Generated if not provided during creation.
        title (str): Human-readable name for the session. Defaults to "New chat"
                    and is automatically updated after the first user query.
        created_at (datetime): UTC timestamp when the session was created.
                              Automatically set to current UTC time.
        updated_at (datetime): UTC timestamp when the session was last modified.
                              Updated each time a new message is added.
        messages (list[dict[str, str]]): List of all messages in the conversation.
                                        Each message dict contains:
                                        - 'role': Either 'user' or 'assistant'
                                        - 'content': The actual message text
                                        - 'created_at': ISO format timestamp
        lock (asyncio.Lock): Async lock for preventing race conditions when
                            multiple requests modify the session simultaneously.
                            Ensures data consistency during concurrent operations.
    
    Example:
        >>> session = ChatSession(session_id="abc123")
        >>> print(session.title)
        'New chat'
        >>> len(session.messages)
        0
    """
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
    """
    Parse a comma-separated string into a clean list of items.
    
    This utility function processes CORS configuration strings where multiple
    values are separated by commas. Each item is stripped of leading/trailing
    whitespace and empty items are filtered out.
    
    Args:
        value (str): A comma-separated string. Example: "http://localhost:3000, http://localhost:8000"
    
    Returns:
        list[str]: A list of trimmed, non-empty strings.
                  Example: ["http://localhost:3000", "http://localhost:8000"]
    
    Example:
        >>> _parse_csv("origin1, origin2, origin3")
        ['origin1', 'origin2', 'origin3']
        >>> _parse_csv("single")
        ['single']
        >>> _parse_csv("  spaced  ,  items  ")
        ['spaced', 'items']
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def _now_utc() -> datetime:
    """
    Get the current UTC time as a timezone-aware datetime object.
    
    This utility function provides a consistent way to get the current time
    for timestamping messages and session updates. Always returns UTC to ensure
    consistent timekeeping across distributed systems.
    
    Returns:
        datetime: Current UTC time with timezone information.
                 Example: 2026-04-20 15:30:45.123456+00:00
    
    Example:
        >>> now = _now_utc()
        >>> print(now.tzinfo)
        UTC
    """
    return datetime.now(timezone.utc)


def _normalize_title(query: str, max_len: int = 72) -> str:
    """
    Convert a user query into a clean, shortened session title.
    
    This function takes a raw user query and processes it to create a suitable
    session title. It removes extra whitespace and truncates long queries to
    prevent UI display issues.
    
    Processing steps:
     1. Normalize whitespace (remove extra spaces and line breaks)
     2. Check if length exceeds maximum
     3. Truncate with ellipsis if needed
    
    Args:
        query (str): The user's original query. Example: "Show me  diamonds"
        max_len (int): Maximum characters for the title. Default is 72.
                      Useful for UI constraints.
    
    Returns:
        str: A normalized, optionally truncated title.
             Examples:
             - "Find me a diamond" -> "Find me a diamond"
             - "<very long query>" -> "<truncated>..."
             - "" -> "New chat"
    
    Example:
        >>> _normalize_title("Show me  round  diamonds")
        'Show me round diamonds'
        >>> _normalize_title("A" * 100)
        'A' * 71 + '...'
    """
    clean = " ".join(query.split())
    if len(clean) <= max_len:
        return clean or "New chat"
    return f"{clean[:max_len - 1]}..."


def _coerce_text(content: object) -> str:
    """
    Convert various content formats into a clean string representation.
    
    This function handles multiple response formats from LLM models which may
    return content as strings, lists, or complex nested structures. It normalizes
    all formats into a single clean string suitable for returning to the user.
    
    Supported input formats:
    - str: Direct string, whitespace trimmed
    - list: Can contain strings or dicts with 'text' or 'content' keys
    - dict: Falls back to str() conversion
    
    Args:
        content (object): Response content from LLM API. Can be any type.
                         Examples:
                         - "Hello world"
                         - ["Hello", "world"]
                         - [{"text": "Hello"}, {"content": "world"}]
    
    Returns:
        str: Cleaned and normalized text. Multiple parts joined by newlines.
             Empty strings and whitespace-only content are filtered out.
             Examples:
             - "Hello" -> "Hello"
             - ["Hello", "world"] -> "Hello\nworld"
    
    Example:
        >>> _coerce_text("  hello  ")
        'hello'
        >>> _coerce_text(["part1", "part2"])
        'part1\npart2'
        >>> _coerce_text([{"text": "hello"}])
        'hello'
    """
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
    """
    Determine if a query requires database lookup via the tool agent.
    
    This function implements a heuristic to decide whether a user query is asking
    about specific diamonds or inventory (requiring database lookup) versus asking
    general questions (which can be answered by the chat model alone).
    
    Query is classified as needing tool lookup if:
    1. It contains diamond-specific keywords (carat, clarity, shape, etc.), OR
    2. It mentions "diamond" AND contains numerical values (prices, carats)
    
    This routing logic optimizes response time by using the fast chat model for
    general questions and only invoking the slower tool agent when necessary.
    
    Args:
        query (str): User's natural language query.
                    Examples:
                    - "Find me a 2 carat diamond" -> True (has carat + digit)
                    - "What is clarity?" -> True (clarity keyword)
                    - "How do I care for diamonds?" -> False (no keywords/numbers)
    
    Returns:
        bool: True if query likely needs database lookup, False otherwise.
    
    Example:
        >>> _requires_tool_lookup("Find a 2 carat round diamond")
        True
        >>> _requires_tool_lookup("What is a diamond?")
        False
        >>> _requires_tool_lookup("Show VS1 diamonds")
        True
    """
    lower_query = query.lower()
    if any(token in lower_query for token in TOOL_QUERY_HINTS):
        return True
    return "diamond" in lower_query and any(ch.isdigit() for ch in lower_query)


def _history_as_messages(
    session: ChatSession,
    max_messages: int,
) -> list[HumanMessage | AIMessage]:
    """
    Convert session message history into LangChain message objects.
    
    This function transforms the simple dict-based message format stored in
    ChatSession into LangChain's message types (HumanMessage/AIMessage) which are
    required by the LLM inference pipeline. It also implements message limiting
    to control context window size and improve performance.
    
    Processing:
    1. Takes the last N messages from session history
    2. Filters out empty messages
    3. Converts role labels ('user'/'assistant') to LangChain message types
    
    Args:
        session (ChatSession): The chat session containing message history.
        max_messages (int): Maximum number of recent messages to include.
                           Used to limit LLM context window. Example: 10
                           (includes last 10 messages)
    
    Returns:
        list[HumanMessage | AIMessage]: List of LangChain message objects
                                        in chronological order.
                                        Empty if session has no messages.
    
    Example:
        >>> session = ChatSession(session_id="123")
        >>> session.messages = [
        ...     {"role": "user", "content": "Hi"},
        ...     {"role": "assistant", "content": "Hello!"},
        ... ]
        >>> messages = _history_as_messages(session, 10)
        >>> len(messages)
        2
        >>> type(messages[0]).__name__
        'HumanMessage'
    """
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
    """
    Convert a ChatSession object into a JSON-serializable dictionary.
    
    This function prepares session data for API responses. It always includes
    session metadata (id, title, timestamps, message count) and optionally
    includes the full message history.
    
    The function converts datetime objects to ISO 8601 format for JSON compatibility.
    This is essential because Python datetime objects cannot be directly serialized
    to JSON.
    
    Args:
        session (ChatSession): The chat session to serialize.
        include_messages (bool): Whether to include the full message list.
                               Default False (metadata only).
                               Set True for detailed session endpoints.
    
    Returns:
        dict: Session data with structure:
              {
                  "session_id": str,
                  "title": str,
                  "created_at": str (ISO 8601),
                  "updated_at": str (ISO 8601),
                  "message_count": int,
                  "messages": list[dict] (optional, if include_messages=True)
              }
    
    Example:
        >>> session = ChatSession(session_id="abc123", title="My Session")
        >>> data = _serialize_session(session)
        >>> data["session_id"]
        'abc123'
        >>> "messages" in data
        False
        >>> data_full = _serialize_session(session, include_messages=True)
        >>> "messages" in data_full
        True
    """
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
    """
    Create a new chat session and register it in the session store.
    
    This function instantiates a new ChatSession, generates a unique ID if needed,
    validates the title, stores it in the in-memory session store, and returns it.
    
    Behavior:
    - If session_id is None, generates a new UUID v4
    - If title is None or empty, defaults to "New chat"
    - Automatically stores the session for later retrieval
    
    Args:
        session_id (str | None): Optional UUID for the session. If None, a new
                                 UUID is generated. Defaults to None.
        title (str | None): Optional display name for the session.
                           Defaults to None (becomes "New chat").
    
    Returns:
        ChatSession: Newly created session object. This session is also added
                    to the global session_store dictionary for retrieval.
    
    Example:
        >>> session = _create_session()
        >>> session.title
        'New chat'
        >>> session.session_id  # Auto-generated UUID
        'a1b2c3d4-...'
        >>> 
        >>> session2 = _create_session(session_id="custom-id", title="My Chat")
        >>> session2.title
        'My Chat'
    """
    resolved_id = session_id or str(uuid.uuid4())
    resolved_title = title.strip() if title and title.strip() else "New chat"
    session = ChatSession(session_id=resolved_id, title=resolved_title)
    session_store[resolved_id] = session
    return session


def _get_or_create_session(session_id: str | None = None) -> ChatSession:
    """
    Retrieve an existing session or create a new one if not found.
    
    This is a convenience function that implements the "get or create" pattern.
    It's useful for endpoints that need to ensure a session exists, whether
    it's a returning user or a first-time request.
    
    Logic:
    1. If session_id is provided and exists in store, return it
    2. Otherwise, create a new session (with provided id or auto-generated)
    
    Args:
        session_id (str | None): Optional session ID to look up.
                                If None, a new session is created.
    
    Returns:
        ChatSession: Either the existing session (if found) or newly created one.
                    Guaranteed to be registered in session_store.
    
    Example:
        >>> session1 = _get_or_create_session()
        >>> session_id = session1.session_id
        >>> session2 = _get_or_create_session(session_id)
        >>> session1 is session2
        True
        >>> session3 = _get_or_create_session("nonexistent")
        >>> session3.session_id
        'nonexistent'
    """
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
    """
    Initialize and return a new NVIDIA chat language model instance.
    
    This factory function creates ChatNVIDIA instances configured for general
    conversation. The chat model is lighter-weight and faster than the tool-calling
    model, making it ideal for answering general questions.
    
    Configuration:
    - Uses settings from utils.config (environment variables)
    - Temperature and top_p control randomness/creativity
    - max_tokens limits response length
    - Streaming is disabled (responses are generated fully before returning)
    
    Returns:
        ChatNVIDIA: Initialized LLM instance ready for inference.
                   Temperature and top_p settings affect response variety.
    
    Note:
        This creates a new instance each time. Use _get_chat_model() to get
        a cached singleton instance for better resource usage.
    
    Example:
        >>> llm = _create_chat_llm()
        >>> response = llm.invoke("What is a diamond?")
        >>> print(response.content)
    """
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
    """
    Create a tool-calling agent for database-backed diamond searches.
    
    This function initializes an agent that can invoke the search_diamonds_db tool
    to query the diamond inventory. It uses a more capable (but slower) LLM model
    optimized for tool use.
    
    The agent uses:
    - A tool-specialized NVIDIA LLM model
    - The search_diamonds_db tool for inventory access
    - A predefined agent_prompt for behavior guidance
    
    Returns:
        Agent: A LangChain agent instance that can dynamically call tools
               based on user queries. Suitable for wrapping in AgentExecutor.
    
    Note:
        This creates a new agent instance each time. Use _get_tool_agent()
        to get a cached singleton for efficiency.
    
    Example:
        >>> agent = _create_tool_agent()
        >>> executor = AgentExecutor(agent=agent, tools=tools)
        >>> result = executor.invoke({"input": "Find 2 carat diamonds"})
    """
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
    """
    Get or initialize the singleton chat LLM instance (lazy initialization).
    
    This function implements lazy initialization with thread-safety using an
    async lock. The first call creates the LLM, subsequent calls return the
    same instance. This pattern is efficient because LLM initialization is
    expensive (API connections, authentication, etc.).
    
    Thread-safety:
    - Uses async lock (chat_model_init_lock) to prevent race conditions
    - Ensures only one LLM instance is created even with concurrent requests
    - Double-checked locking pattern for performance
    
    Returns:
        ChatNVIDIA: Singleton chat LLM instance, the same one across all calls.
    
    Example:
        >>> model = await _get_chat_model()
        >>> model2 = await _get_chat_model()
        >>> model is model2
        True
    """
    global chat_llm
    if chat_llm is not None:
        return chat_llm

    async with chat_model_init_lock:
        if chat_llm is None:
            chat_llm = _create_chat_llm()
    return chat_llm


async def _get_tool_agent():
    """
    Get or initialize the singleton tool-calling agent (lazy initialization).
    
    Similar to _get_chat_model(), this implements lazy initialization with
    async thread-safety. The agent is expensive to initialize so we reuse
    the same instance across all requests.
    
    Thread-safety:
    - Uses async lock (tool_agent_init_lock) to prevent race conditions
    - Ensures only one agent instance is created even with concurrent requests
    - Double-checked locking pattern for performance
    
    Returns:
        Agent: Singleton tool-calling agent instance for database queries.
    
    Example:
        >>> agent = await _get_tool_agent()
        >>> agent2 = await _get_tool_agent()
        >>> agent is agent2
        True
    """
    global tool_agent
    if tool_agent is not None:
        return tool_agent

    async with tool_agent_init_lock:
        if tool_agent is None:
            tool_agent = _create_tool_agent()
    return tool_agent


async def _run_chat_path(query: str, session: ChatSession) -> str:
    """
    Execute the fast chat inference path (no tool calling).
    
    This function handles general questions and conversations that don't require
    database lookups. It uses the lighter-weight chat model for faster responses.
    
    Process:
    1. Get the singleton chat LLM model
    2. Build message history from session (limited to MEMORY_MAX_MESSAGES)
    3. Construct final message list: [system_prompt, history..., user_query]
    4. Call LLM with timeout protection
    5. Extract and clean response text
    
    Args:
        query (str): User's question. Example: "What is clarity?"
        session (ChatSession): Current chat session containing message history.
    
    Returns:
        str: LLM's response text. Guaranteed non-empty (fallback message if failed).
             Example: "Clarity refers to the absence of inclusions..."
    
    Raises:
        asyncio.TimeoutError: If LLM doesn't respond within CHAT_TIMEOUT_SECONDS.
                             Should be caught by caller for graceful degradation.
    
    Example:
        >>> session = _create_session()
        >>> response = await _run_chat_path("Hello!", session)
        >>> len(response) > 0
        True
    """
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
    """
    Execute the tool-calling agent path for database-backed queries.
    
    This function handles queries that need database lookups (searching for
    specific diamonds, inventory checks, etc.). It uses the tool-calling agent
    which can dynamically invoke the search_diamonds_db tool.
    
    Process:
    1. Get the singleton tool-calling agent
    2. Build message history from session
    3. Create AgentExecutor with tool definitions and configuration
    4. Invoke agent with user query and conversation history
    5. Extract response with timeout protection
    
    Args:
        query (str): User's query requiring database lookup.
                    Example: "Find 2 carat VS1 diamonds under $10k"
        session (ChatSession): Current session with message history.
    
    Returns:
        str: Agent's final response text with search results or recommendations.
             Guaranteed non-empty (fallback message if failed).
    
    Raises:
        asyncio.TimeoutError: If agent exceeds AGENT_TIMEOUT_SECONDS.
                             Should be caught by caller for graceful degradation.
    
    Note:
        - Agent can iterate multiple times to refine results
        - Limited by AGENT_MAX_ITERATIONS to prevent infinite loops
        - Parsing errors are handled gracefully (agent continues)
    
    Example:
        >>> session = _create_session()
        >>> response = await _run_tool_path("Find 2 carat diamonds", session)
        >>> "diamond" in response.lower()
        True
    """
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
    """
    Health check endpoint.
    
    Simple endpoint that verifies the API is running and accessible.
    Used by monitoring systems and clients to confirm server availability.
    
    Returns:
        dict: Welcome message. Example: {"message": "Welcome to the Diamond Query API"}
    
    Example:
        GET /
        Response: {"message": "Welcome to the Diamond Query API"}
    """
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the Diamond Query API"}


@app.get("/sessions")
async def list_sessions() -> dict:
    """
    Retrieve a list of all chat sessions.
    
    Returns summaries of all active sessions sorted by most recent update.
    Useful for displaying session history in the UI.
    
    Returns:
        dict: API response containing:
              - status (str): Always "success"
              - sessions (list[dict]): Array of session summaries (metadata only,
                                      not including message content). Sorted by
                                      updated_at descending (most recent first).
    
    Example:
        GET /sessions
        Response:
        {
            "status": "success",
            "sessions": [
                {
                    "session_id": "uuid-123",
                    "title": "My Session",
                    "created_at": "2026-04-20T10:30:00+00:00",
                    "updated_at": "2026-04-20T11:45:00+00:00",
                    "message_count": 5
                }
            ]
        }
    """
    sessions = sorted(session_store.values(), key=lambda item: item.updated_at, reverse=True)
    return {"status": "success", "sessions": [_serialize_session(session) for session in sessions]}


@app.post("/sessions")
async def create_session(request: CreateSessionRequest | None = None) -> dict:
    """
    Create a new chat session.
    
    Initializes a new session with optional user-provided title. If no title
    is provided, defaults to "New chat". The session ID is auto-generated as a UUID.
    
    Args:
        request (CreateSessionRequest | None): Optional request containing:
                                              - title (str | None): Custom session name
    
    Returns:
        dict: API response containing:
              - status (str): Always "success"
              - session (dict): Newly created session metadata
                              (includes session_id, title, timestamps, message_count)
    
    Example:
        POST /sessions
        Request body: {"title": "Diamond Search"}
        Response:
        {
            "status": "success",
            "session": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Diamond Search",
                "created_at": "2026-04-20T12:00:00+00:00",
                "updated_at": "2026-04-20T12:00:00+00:00",
                "message_count": 0
            }
        }
    """
    session = _create_session(title=request.title if request else None)
    return {"status": "success", "session": _serialize_session(session)}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """
    Retrieve a specific session with full message history.
    
    Fetches complete session data including all messages. Used when the client
    needs to load a previous conversation or display chat history.
    
    Args:
        session_id (str): UUID of the session to retrieve.
    
    Returns:
        dict: API response containing:
              - status (str): Always "success"
              - session (dict): Full session data including:
                              - session_id, title, timestamps
                              - message_count: Total number of messages
                              - messages (list[dict]): All messages with:
                                * role: "user" or "assistant"
                                * content: Message text
                                * created_at: ISO 8601 timestamp
    
    Raises:
        HTTPException: 404 if session_id doesn't exist.
    
    Example:
        GET /sessions/550e8400-e29b-41d4-a716-446655440000
        Response:
        {
            "status": "success",
            "session": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Diamond Search",
                "created_at": "2026-04-20T12:00:00+00:00",
                "updated_at": "2026-04-20T12:05:00+00:00",
                "message_count": 2,
                "messages": [
                    {"role": "user", "content": "Find 2 carat diamonds", ...},
                    {"role": "assistant", "content": "I found these...", ...}
                ]
            }
        }
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "session": _serialize_session(session, include_messages=True)}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """
    Delete a chat session and its message history.
    
    Permanently removes a session from the in-memory store. This operation
    cannot be undone (data is not persisted to backup).
    
    Args:
        session_id (str): UUID of the session to delete.
    
    Returns:
        dict: API response containing:
              - status (str): Always "success"
              - session_id (str): ID of the deleted session (confirmation)
    
    Raises:
        HTTPException: 404 if session_id doesn't exist.
    
    Example:
        DELETE /sessions/550e8400-e29b-41d4-a716-446655440000
        Response:
        {
            "status": "success",
            "session_id": "550e8400-e29b-41d4-a716-446655440000"
        }
    """
    removed = session_store.pop(session_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "session_id": session_id}


@app.post("/query")
async def query_diamond(request: DiamondQueryRequest, fastapi_request: Request) -> JSONResponse:
    """
    Main endpoint for processing user diamond queries.
    
    This is the core endpoint that orchestrates the entire query flow:
    1. Determines if query needs database lookup or just chat
    2. Routes to appropriate inference path (chat or tool agent)
    3. Handles timeouts and errors gracefully
    4. Saves messages to session history
    5. Returns comprehensive response with metadata
    
    Session handling:
    - Uses session_id from request body, x-session-id header, or creates new
    - Uses async lock to prevent race conditions on concurrent requests
    - Automatically normalizes session title after first message
    
    Routing logic:
    - Queries with diamond keywords or numeric values -> tool_agent (database lookup)
    - General questions -> chat_fast (faster, no database)
    
    Error handling:
    - Timeouts: Returns user-friendly degraded response
    - Other errors: Logs full exception, returns graceful fallback
    - Status field indicates "success" or "degraded" for monitoring
    
    Args:
        request (DiamondQueryRequest): Query request containing:
                                      - query (str): User's question/request
                                      - session_id (str | None): Optional session UUID
        fastapi_request (Request): FastAPI request object for header access.
                                  Used to read x-session-id header.
    
    Returns:
        JSONResponse: API response with structure:
                      {
                          "session_id": str (UUID),
                          "user_query": str (echoed back),
                          "summary": str (LLM response),
                          "status": str ("success" or "degraded"),
                          "mode": str ("chat_fast" or "tool_agent"),
                          "latency_ms": int (request processing time),
                          "session": dict (updated session metadata)
                      }
                      Also sets x-session-id header for client tracking.
    
    Example:
        POST /query
        Request: {"query": "Find 2 carat round diamonds", "session_id": "abc-123"}
        Response:
        {
            "session_id": "abc-123",
            "user_query": "Find 2 carat round diamonds",
            "summary": "I found 5 excellent options...",
            "status": "success",
            "mode": "tool_agent",
            "latency_ms": 2341,
            "session": {...}
        }
        Headers: x-session-id: abc-123
    
    Note:
        - Large queries may trigger tool_agent path and take longer
        - Messages are always saved, even if response is degraded
        - Session history is maintained in-memory (lost on restart)
    """
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
