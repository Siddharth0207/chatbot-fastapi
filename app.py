from fastapi import FastAPI, Depends
from pydantic import BaseModel , Field
import pandas as pd
import ast
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse




from main import DiamondFinder

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from langchain_community.utilities import SQLDatabase

finder = DiamondFinder("postgresql+asyncpg://postgres:0207@localhost:5432/postgres")



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
    return await {"message": "Welcome to the Diamond Query API"}

@app.post("/query")
async def query_diamond(request: DiamondQueryRequest):
    result = await finder.find_diamonds(request.query)
    return JSONResponse(content=result)
