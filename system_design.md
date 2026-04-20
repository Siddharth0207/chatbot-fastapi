# System Design: AI Diamond Consultant Chatbot

## Overview

This document provides a comprehensive system design for the FastAPI-based diamond recommendation chatbot powered by LangChain and NVIDIA AI endpoints. The system enables users to search for diamonds through natural language conversations with persistent session memory.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│                    (Frontend / Web UI)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ HTTP/REST
                         │
┌─────────────────────────▼────────────────────────────────────────┐
│                    FastAPI Server (app.py)                        │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ REST Endpoints                                               │ │
│ │ • GET  /              - Welcome message                      │ │
│ │ • GET  /sessions      - List all sessions                    │ │
│ │ • POST /sessions      - Create new session                   │ │
│ │ • GET  /sessions/{id} - Get session with messages            │ │
│ │ • DELETE /sessions/{id} - Delete session                     │ │
│ │ • POST /query         - Process query with AI agent          │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Session Management                                           │ │
│ │ • session_store: Dict[session_id -> ChatSession]             │ │
│ │ • ChatSession: Stores memory, messages, metadata             │ │
│ │ • asyncio.Lock: Thread-safe concurrent access               │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ LangChain Agent Orchestration                                │ │
│ │ • LLM: NVIDIA ChatNVIDIA (Gemma-4-31b)                       │ │
│ │ • Agent: Tool-calling agent with search_diamonds_db tool     │ │
│ │ • Memory: ConversationBufferMemory per session               │ │
│ │ • Executor: AgentExecutor for async invocation              │ │
│ └──────────────────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├──────────────────────────┬──────────────────┐
                     │                          │                  │
        ┌────────────▼──────────────┐  ┌───────▼─────────┐  ┌─────▼──────────┐
        │  LangChain Tool Layer      │  │  Utils Layer    │  │  Config Layer  │
        │  (main.py)                 │  │ (utils/*.py)    │  │ (utils/config) │
        │                            │  │                 │  │                │
        │ • DiamondDatabase          │  │ • Logging       │  │ • Settings     │
        │ • normalize_fields()       │  │ • Exceptions    │  │ • CORS config  │
        │ • build_sql_query()        │  │ • Mappings      │  │ • API keys     │
        │ • query_diamonds()         │  │ • Prompts       │  │ • DB URL       │
        │ • search_diamonds_db tool  │  │                 │  │                │
        └────────────┬──────────────┘  └─────────────────┘  └────────────────┘
                     │
        ┌────────────▼──────────────┐
        │   PostgreSQL Database      │
        │                            │
        │ • diamonds table           │
        │ • Async connections        │
        │ • SQLAlchemy ORM           │
        └────────────────────────────┘
```

### Core Components

1. **FastAPI Application Server** (`app.py`)
   - REST API endpoints for session and query management
   - CORS middleware configuration
   - Session state management with in-memory storage
   - Request/response validation with Pydantic

2. **LangChain Agent** (`app.py` + `main.py`)
   - NVIDIA Gemma-4 LLM integration
   - Tool-calling agent for diamond searches
   - Conversation memory per session
   - Async executor for non-blocking operations

3. **Database Layer** (`main.py`)
   - Async PostgreSQL connection pool
   - Field normalization with mappings
   - Dynamic SQL query generation
   - Result formatting and counting

4. **Configuration & Utilities** (`utils/`)
   - Environment-based settings management
   - Structured exception handling
   - Centralized logging with timestamped files
   - Diamond attribute mappings for normalization
   - Agent system prompt definition

---

## Low-Level Architecture & Data Flow

### 1. Session Management Flow

```
Client Request with optional session_id
    │
    └─► query_diamond() endpoint [app.py: L156]
        │
        ├─► Extract session_id from:
        │   1. Request body (DiamondQueryRequest.session_id)
        │   2. x-session-id header
        │
        └─► _get_or_create_session(resolved_session_id) [app.py: L105]
            │
            ├─► Check session_store dict [app.py: L130]
            │
            └─► If not found: _create_session() [app.py: L97]
                │
                ├─► Generate UUID if no session_id provided
                ├─► Create ChatSession dataclass instance [app.py: L66]
                │   - Initializes ConversationBufferMemory [app.py: L71]
                │   - Creates asyncio.Lock for concurrency [app.py: L72]
                │   - Empty messages list
                │   - Timestamps in UTC
                │
                └─► Store in session_store dict [app.py: L100]
```

**Key Code References:**
- Session dataclass: [app.py](app.py#L64-L73)
- Session creation: [app.py](app.py#L97-L101)
- Session retrieval: [app.py](app.py#L104-L108)

### 2. Query Processing Flow

```
POST /query with DiamondQueryRequest
    │
    └─► query_diamond() [app.py: L156]
        │
        ├─► Acquire session.lock [app.py: L167] (async lock for thread safety)
        │
        ├─► Create AgentExecutor [app.py: L168]
        │   └─► Components:
        │       • agent: Tool-calling agent
        │       • tools: [search_diamonds_db] [app.py: L143]
        │       • memory: Session's ConversationBufferMemory [app.py: L71]
        │       • verbose: True for logging
        │
        ├─► agent_executor.ainvoke({"input": request.query}) [app.py: L169]
        │   │
        │   └─► LangChain Agent Loop:
        │       1. Use LLM to understand query
        │       2. Decide if search_diamonds_db tool needed
        │       3. Extract parameters: carat, shape, clarity, etc.
        │       4. Invoke tool with DiamondQuery schema [main.py: L18]
        │       5. Format tool response
        │       6. Generate final assistant response
        │
        ├─► Extract summary from result [app.py: L170]
        │
        ├─► Store message pair in session:
        │   • User message: {role, content, created_at} [app.py: L174-175]
        │   • Assistant message: {role, content, created_at} [app.py: L177-178]
        │
        ├─► Update session timestamps [app.py: L179]
        │
        ├─► Set session title from first query [app.py: L180-181]
        │   └─► _normalize_title() trims and truncates [app.py: L47]
        │
        └─► Return JSONResponse with:
            • session_id (also in header)
            • user_query
            • summary
            • status
            • updated session metadata
```

**Key Code References:**
- Query endpoint: [app.py](app.py#L156-L200)
- Agent initialization: [app.py](app.py#L139-L143)
- Message storage: [app.py](app.py#L174-L178)

### 3. Database Query Flow

```
Agent calls search_diamonds_db(**kwargs) [main.py: L162]
    │
    └─► search_diamonds_db() tool [main.py: L162-L182]
        │
        ├─► get_db() - Get singleton DiamondDatabase [main.py: L148-154]
        │
        ├─► Filter None values from kwargs [main.py: L168]
        │   └─► Remove unspecified search parameters
        │
        ├─► normalize_diamond_fields(active_kwargs) [main.py: L71-82]
        │   │
        │   └─► For each field in MAPPINGS [utils/mappings.py]
        │       ├─► Convert user input (e.g., "EX", "Excellent") to standard (e.g., "EX")
        │       ├─► Handle case variations and abbreviations
        │       └─► Pass through unmapped values unchanged
        │
        ├─► build_sql_query_from_json(normalized) [main.py: L84-130]
        │   │
        │   └─► Build WHERE clause conditions:
        │       ├─► Carat: range, less-than, greater-than, or exact
        │       ├─► String fields (shape, color, lab, etc.): Case-insensitive LIKE
        │       ├─► Boolean fields (heart_and_arrow, eye_clean): Exact match
        │       └─► All conditions joined with AND
        │
        │   └─► Result: "SELECT * FROM diamonds WHERE [conditions] LIMIT 10;"
        │
        ├─► Log SQL query [main.py: L176]
        │
        ├─► query_diamonds(sql) [main.py: L132-145]
        │   │
        │   ├─► Extract WHERE clause for COUNT query [main.py: L134-141]
        │   │
        │   ├─► Execute async database queries [main.py: L143-151]
        │   │   ├─► Main query: Get matching diamonds (max 10)
        │   │   ├─► Count query: Get total matches
        │   │   └─► Convert rows to dict format
        │   │
        │   └─► Return (diamonds: List[Dict], total_count: int)
        │
        └─► Format results as human-readable string [main.py: L177-182]
            ├─► "Found {total_count} diamonds matching criteria..."
            └─► List top 10 with shape, color, cut, clarity, carat, price
```

**Key Code References:**
- Diamond query schema: [main.py](main.py#L18-40)
- Field normalization: [main.py](main.py#L71-82)
- SQL query builder: [main.py](main.py#L84-130)
- Database query execution: [main.py](main.py#L132-145)
- Tool definition: [main.py](main.py#L162-182)

### 4. Configuration & Utilities Flow

```
Application Startup
    │
    ├─► get_settings() [utils/config.py: L37-42]
    │   │
    │   └─► @lru_cache() - Creates singleton Settings instance
    │       ├─► Load .env file
    │       ├─► DATABASE_URL: PostgreSQL async connection string
    │       ├─► NVIDIA_API_KEY: For Gemma LLM access
    │       ├─► CORS_ALLOW_ORIGINS: Comma-separated list
    │       ├─► CORS_ALLOW_ORIGIN_REGEX: Pattern for dynamic origins
    │       └─► HOST/PORT: Server binding configuration
    │
    ├─► Logging initialization [utils/logger.py]
    │   ├─► Create logs directory
    │   ├─► Generate timestamped log file: MM_DD_YYYY_HH_MM_SS.log
    │   └─► Configure basicConfig with format and level
    │
    ├─► Diamond attribute mappings [utils/mappings.py]
    │   └─► MAPPINGS dict with normalizations for:
    │       ├─► symmetry: "EX", "VG", "G", "F", "P", "U"
    │       ├─► cut, polish: Same as symmetry
    │       ├─► clarity: "FL", "IF", "VVS1", "VVS2", "VS1", "VS2", etc.
    │       ├─► color: "D" through "YZ"
    │       ├─► fluorescence: "None", "Faint", "Medium", "Strong"
    │       ├─► shape: "Round", "Oval", "Emerald", "Heart", etc.
    │       ├─► culet: "None", "VS", "S", "M", "L", "EL", "VL"
    │       ├─► lab: "IOD", "GIA", "HRD", "IGI"
    │       └─► eye_clean, heart_and_arrow: "YES", "NO"
    │
    ├─► Agent prompt definition [utils/prompts.py]
    │   └─► AGENT_SYSTEM_PROMPT: Instructs LLM to act as gemologist
    │       ├─► Guidelines for tool usage
    │       ├─► Response formatting rules
    │       └─► Tone and style specifications
    │
    └─► Exception handling [utils/exception.py]
        ├─► error_message_detail(): Extracts traceback info
        └─► CustomException: Wraps errors with file/line context
```

**Key Code References:**
- Settings class: [utils/config.py](utils/config.py#L13-32)
- get_settings function: [utils/config.py](utils/config.py#L35-42)
- Logging setup: [utils/logger.py](utils/logger.py)
- Mappings structure: [utils/mappings.py](utils/mappings.py#L7)
- Agent prompt: [utils/prompts.py](utils/prompts.py#L11-28)

### 5. Concurrent Request Handling

```
Multiple simultaneous POST /query requests
    │
    ├─► query_diamond() request 1 ──┐
    ├─► query_diamond() request 2 ──┼─► Multiple sessions or same session
    └─► query_diamond() request n ──┘
        │
        └─► For same session:
            ├─► Acquire asyncio.Lock [app.py: L167]
            │   └─► await session.lock
            │
            ├─► Execute AgentExecutor.ainvoke() [app.py: L169]
            │   └─► Async, non-blocking LLM call
            │
            ├─► Update messages and timestamps [app.py: L174-179]
            │
            └─► Release lock (exit async context)
                └─► Next waiting request acquires lock
        
        For different sessions:
        ├─► Each has own ChatSession instance
        ├─► Own ConversationBufferMemory
        ├─► Own asyncio.Lock
        └─► Execute in parallel without blocking
```

**Key Code References:**
- Async lock usage: [app.py](app.py#L167)
- ChatSession dataclass with lock: [app.py](app.py#L64-L73)

---

## Data Models & Schemas

### 1. Request Models (Pydantic)

```python
# [app.py: L53-60] - Diamond Search Query Request
class DiamondQueryRequest(BaseModel):
    query: str              # Required, min 1 char
    session_id: str | None  # Optional, uses header if omitted

# [app.py: L63-67] - Create Session Request
class CreateSessionRequest(BaseModel):
    title: str | None       # Optional display title
```

### 2. Data Structures

```python
# [app.py: L64-73] - Chat Session
@dataclass
class ChatSession:
    session_id: str                          # Unique identifier
    title: str = "New chat"                  # Display title
    created_at: datetime                     # Creation timestamp (UTC)
    updated_at: datetime                     # Last update timestamp (UTC)
    memory: ConversationBufferMemory         # LangChain conversation memory
    messages: list[dict[str, str]]           # [{role, content, created_at}]
    lock: asyncio.Lock                       # Thread-safe access control

# [main.py: L18-40] - Diamond Query Parameters (Tool Schema)
class DiamondQuery(BaseModel):
    carat: Optional[float]          # Weight in carats
    shape: Optional[str]            # Round, Oval, Emerald, etc.
    clarity: Optional[str]          # FL, IF, VVS1, VVS2, VS1, VS2, SI1, etc.
    lab: Optional[str]              # GIA, IGI, HRD, IOD, etc.
    symmetry: Optional[str]         # EX, VG, G, F, P, U
    fluorescence: Optional[str]     # None, Faint, Medium, Strong
    heart_and_arrow: Optional[bool] # Boolean flag
    eye_clean: Optional[bool]       # Boolean flag
    culet: Optional[str]            # None, VS, S, M, L, EL, VL
    cut: Optional[str]              # EX, VG, G, F, P, U, NA
    polish: Optional[str]           # EX, VG, G, F, P, U
    color: Optional[str]            # D, E, F, G, H, I, J, K, L, M, N, OP, QR, ST, UV, WX, YZ
```

### 3. Response Models

```python
# GET /sessions Response
{
    "status": "success",
    "sessions": [
        {
            "session_id": str,
            "title": str,
            "created_at": str (ISO format),
            "updated_at": str (ISO format),
            "message_count": int,
            # messages only included with include_messages=True
        }
    ]
}

# POST /query Response (Success)
{
    "session_id": str,
    "user_query": str,
    "summary": str,              # AI response
    "status": "success",
    "session": {...},            # Serialized session
}
# Header: x-session-id: {session_id}

# POST /query Response (Error)
{
    "error": str,
    "status": "error",
    "session_id": str,
}
# Status Code: 500
# Header: x-session-id: {session_id}
```

---

## Technology Stack

| Layer | Technology | Purpose | Reference |
|-------|-----------|---------|-----------|
| **Framework** | FastAPI | REST API server | [app.py](app.py#L1) |
| **LLM** | NVIDIA ChatNVIDIA | Gemma-4-31b model | [app.py](app.py#L135-143) |
| **Agent Framework** | LangChain | Tool-calling agent, memory | [app.py](app.py#L143-145) |
| **Database** | PostgreSQL | Diamond records storage | [main.py](main.py#L60) |
| **Async ORM** | SQLAlchemy | Async database queries | [main.py](main.py#L55-57) |
| **Config** | Pydantic Settings | Environment management | [utils/config.py](utils/config.py#L13) |
| **Validation** | Pydantic | Request/response schemas | [app.py](app.py#L53), [main.py](main.py#L18) |
| **Logging** | Python logging | Application logging | [utils/logger.py](utils/logger.py) |
| **CORS** | FastAPI CORSMiddleware | Cross-origin requests | [app.py](app.py#L127-133) |

---

## Key Design Patterns

### 1. Singleton Pattern
- **Database Instance**: Single DiamondDatabase per application
  ```python
  _db_instance = None
  def get_db() -> DiamondDatabase:  # [main.py: L148-154]
  ```
- **Settings Instance**: Cached via `@lru_cache()`
  ```python
  @lru_cache()
  def get_settings() -> Settings:  # [utils/config.py: L35-42]
  ```

### 2. Session Management Pattern
- **In-memory Session Store**: Dict mapping session_id to ChatSession
  ```python
  session_store: dict[str, ChatSession] = {}  # [app.py: L130]
  ```
- **Per-session Memory**: Each session has isolated ConversationBufferMemory
  ```python
  memory: ConversationBufferMemory  # [app.py: L71]
  ```

### 3. Async/Await Pattern
- **Non-blocking Operations**: All database and LLM calls are async
  ```python
  async def query_diamonds(self, sql: str) -> Tuple[...]:  # [main.py: L132]
  async def ainvoke(...):  # [app.py: L169]
  ```
- **Lock-based Concurrency**: asyncio.Lock prevents race conditions in session updates
  ```python
  async with session.lock:  # [app.py: L167]
  ```

### 4. Tool-calling Agent Pattern
- **LangChain Tool**: Diamond search exposed as callable tool to agent
  ```python
  @tool("search_diamonds_db", args_schema=DiamondQuery)  # [main.py: L162]
  async def search_diamonds_db(**kwargs) -> str:
  ```
- **Dynamic Tool Invocation**: Agent decides when/how to call tool based on user input
  ```python
  tools = [search_diamonds_db]  # [app.py: L143]
  agent = create_tool_calling_agent(llm, tools, agent_prompt)  # [app.py: L144]
  ```

### 5. Normalization & Mapping Pattern
- **Field Normalization**: User input mapped to database values
  ```python
  def normalize_diamond_fields(self, data: dict) -> dict:  # [main.py: L71-82]
      # Uses MAPPINGS dict [utils/mappings.py]
  ```

---

## Security & Error Handling

### 1. Validation
- **Pydantic Models**: Automatic validation on all inputs
  - DiamondQueryRequest: Enforces min_length=1 on query [app.py: L57]
  - DiamondQuery: Optional fields with descriptions [main.py: L18-40]

### 2. CORS Configuration
- **Whitelist-based**: Explicit origins allowed
  ```python
  allow_origins=cors_origins  # [app.py: L128]
  allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX  # [app.py: L129]
  ```

### 3. Error Handling
- **Custom Exception**: Enhanced error context
  ```python
  class CustomException(Exception):  # [utils/exception.py: L24-40]
      # Includes file name, line number, error details
  ```
- **Try-catch at Agent Level**: Handles LLM/tool failures gracefully
  ```python
  try:  # [app.py: L170]
      result = await agent_executor.ainvoke(...)
  except Exception as exc:  # [app.py: L190]
      # Return error response with status 500
  ```

### 4. Logging
- **Centralized Logging**: All operations logged with timestamps
  ```python
  logging.info(f"Query successful for session {session.session_id}")  # [app.py: L184]
  logging.error(f"Error in /query endpoint...", exc_info=True)  # [app.py: L191]
  ```

---

## API Endpoints Documentation

### 1. GET /
Welcome endpoint
```
Response: {"message": "Welcome to the Diamond Query API"}
```
[Code Reference: app.py](app.py#L111-114)

### 2. GET /sessions
List all chat sessions
```
Response: {"status": "success", "sessions": [...]}
```
[Code Reference: app.py](app.py#L117-121)

### 3. POST /sessions
Create new chat session
```
Request: {title: str | None}
Response: {"status": "success", "session": {...}}
```
[Code Reference: app.py](app.py#L124-129)

### 4. GET /sessions/{session_id}
Get session with message history
```
Response: {"status": "success", "session": {..., "messages": [...]}}
```
[Code Reference: app.py](app.py#L132-140)

### 5. DELETE /sessions/{session_id}
Delete a session
```
Response: {"status": "success", "session_id": str}
```
[Code Reference: app.py](app.py#L143-151)

### 6. POST /query
Process diamond search query with AI agent
```
Request: {query: str, session_id: str | None}
Header: x-session-id: {session_id} (optional)
Response: {session_id, user_query, summary, status, session}
Header: x-session-id: {session_id}
```
[Code Reference: app.py](app.py#L154-200)

---

## Database Schema (Diamond Table)

Expected PostgreSQL table structure:
```sql
CREATE TABLE diamonds (
    id SERIAL PRIMARY KEY,
    shape VARCHAR(50),              -- Round, Oval, Emerald, etc.
    color CHAR(2),                  -- D, E, F, G, H, I, J, K, etc.
    cut VARCHAR(20),                -- EX, VG, G, F, P, U
    clarity VARCHAR(10),            -- FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, SI3, I1, I2, I3
    carat DECIMAL(10, 2),           -- Weight in carats
    symmetry VARCHAR(20),           -- EX, VG, G, F, P, U
    fluorescence VARCHAR(20),       -- None, Faint, Medium, Strong
    heart_and_arrow BOOLEAN,        -- True/False
    eye_clean BOOLEAN,              -- True/False
    culet VARCHAR(20),              -- None, VS, S, M, L, EL, VL
    polish VARCHAR(20),             -- EX, VG, G, F, P, U
    lab VARCHAR(50),                -- GIA, IGI, HRD, IOD, etc.
    price_per_carat DECIMAL(10, 2), -- Price per carat
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Query references this table:
- [main.py: L84-130] - SQL query builder uses "diamonds" table
- [main.py: L134-141] - COUNT query against diamonds table

---

## Deployment Considerations

### Environment Variables (.env)
```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/db_name
NVIDIA_API_KEY=your_api_key_here
HOST=0.0.0.0
PORT=8000
CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173,...
```

### Dependencies
```txt
fastapi           # Web framework [app.py: L8]
uvicorn           # ASGI server
sqlalchemy        # Async ORM [main.py: L8]
asyncpg           # PostgreSQL async driver
langchain         # Agent framework [app.py: L11-12]
langchain-nvidia  # NVIDIA integration [app.py: L13]
pydantic          # Data validation [app.py: L10]
python-dotenv     # Environment loading [utils/config.py: L2]
```

### Scaling Notes
- **Stateful Sessions**: Current in-memory session_store doesn't persist across server restarts. For production, migrate to Redis or database-backed sessions.
- **Database Connections**: SQLAlchemy async connection pool configurable via DATABASE_URL parameters.
- **LLM Rate Limiting**: NVIDIA API may have rate limits; implement retry logic if needed.

---

## Summary

This diamond recommendation chatbot implements a modern AI-powered backend combining:

1. **Conversational AI**: LangChain agent with NVIDIA Gemma-4 LLM
2. **Tool Integration**: Dynamic tool calling for database searches
3. **Session Persistence**: Per-user conversation memory and message history
4. **Async Architecture**: Non-blocking operations for scalability
5. **Data Normalization**: Intelligent field mapping for flexible user inputs
6. **RESTful API**: Clean endpoints for session and query management
7. **Production-Ready**: Error handling, logging, validation, and CORS support

The architecture supports concurrent users with isolated sessions, automatic database query generation from natural language, and AI responses based on retrieved diamond data.
