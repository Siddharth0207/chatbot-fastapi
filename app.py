"""
This module defines a FastAPI application for querying diamond data using a natural language interface.
It integrates LangChain for conversational memory and NVIDIA AI endpoints for processing queries.
"""
from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory

from main import search_diamonds_db
from utils.config import get_settings
from utils.logger import logging
from utils.prompts import agent_prompt

def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

class DiamondQueryRequest(BaseModel):
    query: str

app = FastAPI()
settings = get_settings()
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

# In-memory store for session memories
session_memories = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return session_memories[session_id]

# Initialize LLM and Agent once
llm = ChatNVIDIA(
    model="google/gemma-4-31b-it",
    task="chat",
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    stream = True,
    api_key=settings.NVIDIA_API_KEY
)

tools = [search_diamonds_db]
agent = create_tool_calling_agent(llm, tools, agent_prompt)


@app.get("/")
async def read_root():
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the Diamond Query API"}


@app.post("/query")
async def query_diamond(request: DiamondQueryRequest, fastapi_request: Request):
    session_id = fastapi_request.headers.get("x-session-id")
    if not session_id:
        session_id = str(uuid.uuid4())
        
    logging.info(f"/query endpoint called. Session: {session_id}, Query: {request.query}")
    
    memory = get_memory(session_id)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    
    try:
        # The agent invokes the tool internally if needed, and formulates a response text
        result = await agent_executor.ainvoke({"input": request.query})
        summary = result.get("output", "")
        
        logging.info(f"Query successful for session {session_id}")
        
        # Format the response for API consumption
        formatted_result = {
            "session_id": session_id,
            "user_query": request.query,
            "summary": summary,
            "status": "success"
        }
        
        return JSONResponse(content=formatted_result, headers={"x-session-id": session_id})
    except Exception as e:
        logging.error(f"Error in /query endpoint for session {session_id}: {e}", exc_info=True)
        return JSONResponse(content={
            "error": str(e), 
            "status": "error",
            "session_id": session_id
        }, status_code=500)
