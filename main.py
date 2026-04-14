"""This module provides functionality for querying a PostgreSQL database for diamond records.
It defines the tool for an agentic framework to interact with the database.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession 
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from langchain_core.tools import tool

from utils.mappings import MAPPINGS
from utils.config import get_settings
from utils.logger import logging
import json

class DiamondQuery(BaseModel):
    """
    Pydantic model representing the structured fields for a diamond search query.
    Used as the argument schema for the search_diamonds_db tool.
    """
    carat: Optional[float] = Field(default=None, description="Carat weight of the diamond")
    shape: Optional[str] = Field(default=None, description="Shape of the diamond like Round, Oval, etc.")
    clarity: Optional[str] = Field(default=None, description="Clarity grade like VVS, VS, SI, etc.")
    lab: Optional[str] = Field(default=None, description="Country of lab or lab like India, USA, etc.")
    symmetry: Optional[str] = Field(default=None, description="Symmetry rating, e.g., Excellent, Very Good, etc.")
    fluorescence: Optional[str] = Field(default=None, description="Fluorescence level, e.g., None, Faint, Medium, Strong")
    heart_and_arrow: Optional[bool] = Field(default=None, description="Whether the diamond exhibits Heart and Arrow pattern")
    eye_clean: Optional[bool] = Field(default=None, description="Whether the diamond is eye-clean (inclusions not visible to the naked eye)")
    culet: Optional[str] = Field(default=None, description="Culet size, e.g., None, Small, Medium, Large")
    cut: Optional[str] = Field(default=None, description="Cut grade like Excellent, Very Good, Good, etc.")
    polish: Optional[str] = Field(default=None, description="Polish rating, e.g., Excellent, Very Good, etc.")
    color: Optional[str] = Field(default=None, description="Color grade like D, E, F, etc.")


class DiamondDatabase:
    """
    Provides methods to normalize extracted fields, build SQL queries, 
    and asynchronously query a PostgreSQL database for matching diamonds.
    """
    def __init__(self, db_url: str | None = None):
        settings = get_settings()
        if db_url is None:
            db_url = settings.DATABASE_URL

        # Use async engine for PostgreSQL
        self.engine = create_async_engine(db_url, future=True)
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    def normalize_diamond_fields(self, data: dict) -> dict:
        normalized = data.copy()
        for field, mapping in MAPPINGS.items():
            if field in data and data[field]:
                val = str(data[field]).strip()
                normalized[field] = mapping.get(val, val)
        return normalized

    def build_sql_query_from_json(self, data: dict) -> str:
        base_query = "SELECT * FROM diamonds WHERE "
        conditions = []
        if data.get("carat_range") and isinstance(data["carat_range"], list) and len(data["carat_range"]) == 2:
            min_carat, max_carat = data["carat_range"]
            conditions.append(f"carat > {min_carat} AND carat <= {max_carat}")
        elif data.get("carat_lt") is not None:
            conditions.append(f"carat < {data['carat_lt']}")
        elif data.get("carat_gt") is not None:
            conditions.append(f"carat > {data['carat_gt']}")
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
            
        return base_query + " AND ".join(conditions) + " LIMIT 10;"

    async def query_diamonds(self, sql: str) -> Tuple[List[Dict[str, Any]], int]:
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
            result = await session.execute(text(sql))
            rows = result.fetchall()
            column_names = result.keys()
            diamonds = [dict(zip(column_names, row)) for row in rows]
            
            count_result = await session.execute(text(count_sql))
            total_count = count_result.scalar() or 0
            
            return diamonds, total_count

# Singleton DB instance for the tool
_db_instance = None

def get_db():
    global _db_instance
    if _db_instance is None:
        _db_instance = DiamondDatabase()
    return _db_instance

@tool("search_diamonds_db", args_schema=DiamondQuery)
async def search_diamonds_db(**kwargs) -> str:
    """Search the PostgreSQL database for diamonds matching the given parameters. Returns a summary string of the matches."""
    db = get_db()
    # Filter out None values to keep the query conditions accurate
    active_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    normalized = db.normalize_diamond_fields(active_kwargs)
    sql = db.build_sql_query_from_json(normalized)
    
    logging.info(f"Executing Agent Tool DB Search: {sql}")
    diamonds, total_count = await db.query_diamonds(sql)
    
    if not diamonds:
        return f"No diamonds found matching the criteria."
    
    result_str = f"Found {total_count} diamonds matching criteria. Here are the top {len(diamonds)}:\n"
    for i, d in enumerate(diamonds):
        result_str += (f"{i+1}. Shape: {d.get('shape')}, Color: {d.get('color')}, "
                       f"Cut: {d.get('cut')}, Clarity: {d.get('clarity')}, "
                       f"Carat: {d.get('carat')}, Price: ${d.get('price_per_carat')}\n")
    
    return result_str