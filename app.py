"""
This module defines a FastAPI application for querying diamond data using a natural language interface.
It integrates LangChain for conversational memory and NVIDIA AI endpoints for processing queries.
Classes:
    DiamondQueryRequest (BaseModel): 
        A Pydantic model representing the structure of the request body for diamond queries.
Functions:
    get_memory(session_id: str) -> ConversationBufferMemory:
        Creates and returns a conversation buffer memory object for maintaining chat history.
    read_root() -> dict:
        A FastAPI route that returns a welcome message for the API.
    query_diamond(request: DiamondQueryRequest, fastapi_request: Request) -> JSONResponse:
        A FastAPI route that processes diamond queries using the DiamondFinder class and returns the results.
FastAPI Application:
    app: 
        The main FastAPI application instance with CORS middleware configured for cross-origin requests.
"""
from fastapi import FastAPI, Depends
from pydantic import BaseModel , Field
import pandas as pd
import ast
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request




from main import DiamondFinder

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from langchain_community.utilities import SQLDatabase
from utils.logger import logging


def get_memory(session_id: str)-> ConversationBufferMemory:
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


class DiamondQueryRequest(BaseModel):
    query: str



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:8000"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the Diamond Query API"}

@app.post("/query")
async def query_diamond(request: DiamondQueryRequest, fastapi_request: Request):
    session_id = fastapi_request.headers.get("x-session-id", "default_session")
    logging.info(f"/query endpoint called. Session: {session_id}, Query: {request.query}")
    finder = DiamondFinder("postgresql+asyncpg://postgres:0207@localhost:5432/postgres")
    try:
        result = await finder.find_diamonds(request.query)
        logging.info(f"Query successful for session {session_id}")
        
        # Format the response for better API consumption
        formatted_result = {
            "session_id": session_id,
            "user_query": request.query,
            "summary": result.get("summary", ""),
            "total_count": result.get("total_count", 0),
            "sql_query": result.get("sql_query", ""),
            "diamonds": result.get("diamonds", []),
            "status": "success"
        }
        
        logging.info(f"Formatted result: {formatted_result}")
        return JSONResponse(content=formatted_result, headers={"x-session-id": session_id})
    except Exception as e:
        logging.error(f"Error in /query endpoint for session {session_id}: {e}", exc_info=True)
        return JSONResponse(content={
            "error": str(e), 
            "status": "error",
            "session_id": session_id
        }, status_code=500)
