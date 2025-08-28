"""This module provides functionality for extracting user preferences for diamonds from natural language queries,
normalizing the extracted fields, building SQL queries, and querying a PostgreSQL database asynchronously.
It is designed for integration with FastAPI backends and supports asyncio for multi-user concurrency.
Classes:
    DiamondQuery:
        A Pydantic model representing the structured fields for a diamond search query.
    DiamondFinder:
        A class that provides methods to:
        - Extract user preferences from natural language queries using an LLM.
        - Normalize the extracted fields to match database values.
        - Build SQL queries from the structured fields.
        - Query a PostgreSQL database asynchronously for matching diamonds.
        - Generate an LLM-based summary of the query results.
Methods:
    DiamondFinder.__init__(db_url: str):
        Initializes the DiamondFinder with an async database engine and LLM setup.
    DiamondFinder.extract_diamond_entities(user_input: str) -> dict:
        Extracts diamond search fields from a user's natural language query using an LLM.
    DiamondFinder.normalize_diamond_fields(data: dict) -> dict:
        Normalizes user/LLM-provided diamond fields to match database values.
    DiamondFinder.build_sql_query_from_json(data: dict) -> str:
        Builds a SQL query string from structured diamond search fields.
    DiamondFinder.query_diamonds(sql: str) -> List[Dict[str, Any]]:
        Executes a SQL query asynchronously and returns the result as a list of dictionaries.
    DiamondFinder.find_diamonds(user_query: str) -> Dict[str, Any]:
        Orchestrates the extraction, normalization, SQL building, and database querying for diamonds.
    DiamondQuery:
        - carat: Optional[float] - Carat weight of the diamond.
        - clarity: Optional[str] - Clarity grade (e.g., VVS, VS, SI, etc.).
        - lab: Optional[str] - Country or lab name (e.g., India, USA, GIA, etc.).
        - symmetry: Optional[str] - Symmetry rating (e.g., Excellent, Very Good, etc.).
        - fluorescence: Optional[str] - Fluorescence level (e.g., None, Faint, Medium, Strong).
        - heart_and_arrow: Optional[bool] - Whether the diamond exhibits Heart and Arrow pattern.
        - eye_clean: Optional[bool] - Whether the diamond is eye-clean (inclusions not visible to the naked eye).
        - culet: Optional[str] - Culet size (e.g., None, Small, Medium, Large).
        - cut: Optional[str] - Cut grade (e.g., Excellent, Very Good, Good, etc.).
        - polish: Optional[str] - Polish rating (e.g., Excellent, Very Good, etc.).
    DiamondFinder:
        - engine: SQLAlchemy async engine for PostgreSQL database connection.
        - async_session: SQLAlchemy async session factory.
        - parser: LangChain PydanticOutputParser for extracting structured fields from LLM output.
        - prompt: LangChain ChatPromptTemplate for LLM extraction prompt.
        - llm: LangChain NVIDIA LLM client for both extraction and summary tasks.
        """
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from utils.prompts import extraction_prompt, SUMMARY_PROMPT
from utils.mappings import MAPPINGS
from utils.config import get_settings
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession 
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from utils.logger import logging
import asyncio
import json
import os
from dotenv import load_dotenv


class DiamondQuery(BaseModel):
    """
    Pydantic model representing the structured fields for a diamond search query.

    Attributes:
        carat (Optional[float]): Carat weight of the diamond.
        clarity (Optional[str]): Clarity grade (e.g., VVS, VS, SI, etc.).
        lab (Optional[str]): Country or lab name (e.g., India, USA, GIA, etc.).
        symmetry (Optional[str]): Symmetry rating (e.g., Excellent, Very Good, etc.).
        fluorescence (Optional[str]): Fluorescence level (e.g., None, Faint, Medium, Strong).
        heart_and_arrow (Optional[bool]): Whether the diamond exhibits Heart and Arrow pattern.
        eye_clean (Optional[bool]): Whether the diamond is eye-clean (inclusions not visible to the naked eye).
        culet (Optional[str]): Culet size (e.g., None, Small, Medium, Large).
        cut (Optional[str]): Cut grade (e.g., Excellent, Very Good, Good, etc.).
        polish (Optional[str]): Polish rating (e.g., Excellent, Very Good, etc.).
    """
# If you do not mention the defualt value, the Pydantic Model will consider it as required. 
    carat: Optional[float] = Field(default=None, description="Carat weight of the diamond")
    shape: Optional[str] = Field(default=None,description="Shape of the diamond like Round, Oval, etc.")
    clarity: Optional[str] = Field(default=None,description="Clarity grade like VVS, VS, SI, etc.")
    lab: Optional[str] = Field(default=None,description="Country of lab or lab like India, USA, etc.")
    symmetry: Optional[str] = Field(default=None,description="Symmetry rating, e.g., Excellent, Very Good, etc.")
    fluorescence: Optional[str] = Field(default=None,description="Fluorescence level, e.g., None, Faint, Medium, Strong")
    heart_and_arrow: Optional[bool] = Field(default=None,description="Whether the diamond exhibits Heart and Arrow pattern")
    eye_clean: Optional[bool] = Field(default=None,description="Whether the diamond is eye-clean (inclusions not visible to the naked eye)")
    culet: Optional[str] = Field(default=None,description="Culet size, e.g., None, Small, Medium, Large")
    cut: Optional[str] = Field(default=None,description="Cut grade like Excellent, Very Good, Good, etc.")
    polish: Optional[str] = Field(default=None,description="Polish rating, e.g., Excellent, Very Good, etc.")
    color: Optional[str] = Field(default=None,description="Color grade like D, E, F, etc.")

class DiamondFinder:
    """
    DiamondFinder provides methods to extract user preferences from natural language queries using an LLM,
    normalize the extracted fields, build SQL queries, and asynchronously query a PostgreSQL database for matching diamonds.
    Supports asyncio for multi-user concurrency and is designed for integration with FastAPI backends.

    Attributes:
        engine: SQLAlchemy async engine for PostgreSQL database connection.
        async_session: SQLAlchemy async session factory.
        parser: LangChain PydanticOutputParser for extracting structured fields from LLM output.
        prompt: LangChain ChatPromptTemplate for LLM extraction prompt.
        llm: LangChain NVIDIA LLM client for both extraction and summary tasks.
    """
    def __init__(self, db_url: str | None = None):
        """
        Initialize DiamondFinder with async database engine and LLM setup.

        Args:
            db_url (Optional[str]): Database connection string (should use asyncpg driver for async).
                                    If not provided, will use value from `config.get_settings()`.
        """
        settings = get_settings()
        if db_url is None:
            db_url = settings.DATABASE_URL

        # Use async engine for PostgreSQL
        self.engine = create_async_engine(db_url, future=True)
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

        # Parser and prompts
        self.parser = PydanticOutputParser(pydantic_object=DiamondQuery)
        self.prompt = extraction_prompt

        # LLM client (uses centralized settings for API key)
        self.llm = ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            task="chat",
            temperature=0.6,
            top_p=0.7,
            max_tokens=4096,
            api_key=settings.NVIDIA_API_KEY
        )

        # Chain with memory
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_parser=self.parser,
            verbose=True  # Optional: for debugging
        )

    def extract_diamond_entities(self, user_input: str) -> dict:
        """
        Use the LLM to extract diamond search fields from a user query.

        Args:
            user_input (str): The user's natural language query describing diamond preferences.

        Returns:
            dict: Dictionary of extracted diamond search fields as defined by DiamondQuery.
        """
        sql_query = self.chain.run(input=user_input, format_instructions=self.parser.get_format_instructions())
        if isinstance(sql_query, dict):
            return sql_query
        elif hasattr(sql_query, "dict"):
            return sql_query.dict()
        else:
            # If it's a string, try to parse it as JSON
            try:
                return json.loads(sql_query)
            except Exception:

                raise ValueError("LLM output is not a valid dict or JSON string")

    def normalize_diamond_fields(self, data: dict) -> dict:
        """
        Normalize user/LLM-provided diamond fields to match database values.

        Args:
            data (dict): Dictionary of diamond search fields (possibly with user/LLM variations).

        Returns:
            dict: Dictionary with normalized field values suitable for SQL/database queries.
        """
        normalized = data.copy()
        for field, mapping in MAPPINGS.items():
            if field in data and data[field]:
                val = str(data[field]).strip()
                normalized[field] = mapping.get(val, val)
        return normalized

    def build_sql_query_from_json(self, data: dict) -> str:
        """
        Build a SQL query string from structured diamond search fields.

        Args:
            data (dict): Dictionary of normalized diamond search fields.

        Returns:
            str: SQL query string to retrieve matching diamonds from the database.
        """
        base_query = "SELECT * FROM diamonds WHERE "
        conditions = []
        # Carat range
        if data.get("carat_range") and isinstance(data["carat_range"], list) and len(data["carat_range"]) == 2:
            min_carat, max_carat = data["carat_range"]
            conditions.append(f"carat > {min_carat} AND carat <= {max_carat}")
        # Carat less than
        elif data.get("carat_lt") is not None:
            conditions.append(f"carat < {data['carat_lt']}")
        # Carat greater than
        elif data.get("carat_gt") is not None:
            conditions.append(f"carat > {data['carat_gt']}")
        # Carat exact
        elif data.get("carat") is not None:
            conditions.append(f"carat = {data['carat']}")

        if data.get("shape"):
            conditions.append(f"shape ILIKE '%{data['shape']}%'")
        if data.get("color"):
            conditions.append(f"color ILIKE '%{data['color']}%'")
        if data.get("lab"):
            conditions.append(f"lab ILIKE '%{data['lab']}%'")
        if data.get("clarity"):
            conditions.append(f"clarity ILIKE '%{data['clarity']}%'")
        if data.get("symmetry"):
            conditions.append(f"symmetry ILIKE '%{data['symmetry']}%'")
        if data.get("fluorescence"):
            conditions.append(f"fluorescence ILIKE '%{data['fluorescence']}%'")
        if data.get("heart_and_arrow") is not None:
            conditions.append(f"heart_and_arrow = {data['heart_and_arrow']}")
        if data.get("eye_clean") is not None:
            conditions.append(f"eye_clean = {data['eye_clean']}" )
        if data.get("culet"):
            conditions.append(f"culet ILIKE '%{data['culet']}%'")
        if data.get("cut"):
            conditions.append(f"cut ILIKE '%{data['cut']}%'")
        if data.get("polish"):
            conditions.append(f"polish ILIKE '%{data['polish']}%'")
        if not conditions:
            return "SELECT * FROM diamonds LIMIT 10;"
        logging.info(f"Built SQL query: {base_query + ' AND '.join(conditions)}")
        return base_query + " AND ".join(conditions) + " LIMIT 10;"
       

    async def query_diamonds(self, sql: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        Execute a SQL query asynchronously and return the result as a list of dictionaries, along with the total count of rows matched (ignoring LIMIT).

        Args:
            sql (str): SQL query string to execute (with LIMIT).

        Returns:
            Tuple[List[Dict[str, Any]], int]:
                - List of dictionaries, each representing a diamond record from the database.
                - Total number of rows matched by the query (ignoring LIMIT).
        """
        # Extract WHERE clause for count query
        sql_lower = sql.lower()
        where_idx = sql_lower.find(" where ")
        limit_idx = sql_lower.find(" limit ")
        if where_idx != -1:
            if limit_idx != -1:
                where_clause = sql[where_idx:limit_idx]
            else:
                where_clause = sql[where_idx:]
            count_sql = f"SELECT COUNT(*) FROM diamonds{where_clause};"
        else:
            count_sql = "SELECT COUNT(*) FROM diamonds;"
        async with self.async_session() as session:
            # Get limited results
            result = await session.execute(text(sql))
            rows = result.fetchall()
            column_names = result.keys()
            diamonds = [dict(zip(column_names, row)) for row in rows]
            # Get total count
            count_result = await session.execute(text(count_sql))
            total_count = count_result.scalar() or 0
            return diamonds, total_count
    
    

    async def find_diamonds(self, user_query: str) -> Dict[str, Any]:
        """
        Orchestrate the extraction, normalization, SQL building, and DB querying for diamonds (async).
        Uses memory only for the summary/chat step.
        Args:
            user_query (str): The user's natural language query describing diamond preferences.
            chat_history (list): List of previous messages (optional, for multi-turn chat).
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'diamonds': List of matching diamond records (as dicts)
                - 'summary': LLM-generated summary of the results (str)
                - 'sql_query': The SQL query string used (str)
                - 'chat_history': Updated chat history (list)
                - 'session_id': The session ID (str)
                - 'total_count': Total number of rows matched by the query (int)
        """
        def format_diamond_list(diamonds: List[Dict[str, Any]]) -> str:
            return "\n".join(
                f"{i+1}. Shape: {d['shape']}, Color: {d['color']}, Cut: {d['cut']}, "
                f"Clarity: {d['clarity']}, Polish: {d['polish']}, "
                f"Weight: {d['carat']}, Price/Carat: ${d['price_per_carat']}"
                for i, d in enumerate(diamonds[:10])
            )
        # Extraction (stateless)
        structured_data = self.extract_diamond_entities(user_query)
        normalized_data = self.normalize_diamond_fields(structured_data)
        sql = self.build_sql_query_from_json(normalized_data)
        diamonds, total_count = await self.query_diamonds(sql)
        


        # Prepare summary prompt using centralized prompt template
        summary_prompt = SUMMARY_PROMPT.format(
            diamonds=format_diamond_list(diamonds),
            user_query=user_query,
            total_count=total_count
        )

        # Run summary LLM statelessly
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.llm.invoke, [HumanMessage(content=summary_prompt)]
        )
        logging.info(f"LLM summary response: {response.content}")
        
        return {
            "diamonds": diamonds,
            "summary": response.content,
            "sql_query": sql,
            "total_count": total_count
        }