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
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession 
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
import asyncio
import json


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
    def __init__(self, db_url: str, memory: Optional[ConversationBufferMemory] = None):
        """
        Initialize DiamondFinder with async database engine and LLM setup.

        Args:
            db_url (str): Database connection string (should use asyncpg driver for async).
        """
        # Use async engine for PostgreSQL
        self.engine = create_async_engine(db_url, future=True)
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        self.memory = ConversationBufferMemory(memory_key="chat_history", 
                                               input_key="input", 
                                               return_messages=True)
        self.parser = PydanticOutputParser(pydantic_object=DiamondQuery)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that extracts diamond preferences into structured fields."),
            ("human", "Extract the diamond query fields from: {input}\n\n{format_instructions}")
        ])
        self.llm = ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            task="chat",
            temperature=0.6,
            top_p=0.7,
            max_tokens=4096,
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
        mappings = {
            "symmetry": {
                # EX
                "Excellent": "EX", "X": "EX", "EXCELLENT": "EX", "Ex": "EX", "EXC": "EX", "EXCL": "EX",
                # VG
                "Very Good": "VG", "V": "VG", "V.Good": "VG", "VG": "VG", "VG-": "VG", "VGOOD": "VG",
                # G
                "Good": "G", "GD": "G", "G": "G",
                # F
                "Fair": "F", "F": "F", "FR": "F",
                # P
                "Poor": "P", "P": "P", "PR": "P",
                # U
                "U": "U", "u": "U", "Unknown": "U", "UNKNOWN": "U"
            },
            "cut": {
                # EX
                "Excellent": "EX", "X": "EX", "EXCELLENT": "EX", "Ex": "EX", "EXC": "EX", "EX+": "EX", "EX-": "EX", "EXCL": "EX", "EX": "EX", "BL": "EX", "bl": "EX", "Black Label": "EX",
                # VG
                "Very Good": "VG", "V": "VG", "V.Good": "VG", "VG": "VG", "VG-": "VG", "VGOOD": "VG",
                # G
                "Good": "G", "GD": "G", "G": "G", "GD+": "G",
                # F
                "Fair": "F", "F": "F", "FR": "F",
                # P
                "Poor": "P", "P": "P", "PR": "P",
                # U
                "U": "U", "u": "U", "Unknown": "U", "UNKNOWN": "U",
                # NA
                "Not Applicable": "NA", "NOT APPLICABLE": "NA", "NA": "NA"
            },
            "polish": {
                # EX
                "Excellent": "EX", "X": "EX", "EXCELLENT": "EX", "Ex": "EX", "EXC": "EX", "EXCL": "EX",
                # VG
                "Very Good": "VG", "V": "VG", "V.Good": "VG", "VG": "VG", "VG-": "VG", "VGOOD": "VG",
                # G
                "Good": "G", "GD": "G", "G": "G",
                # F
                "Fair": "F", "F": "F", "FR": "F",
                # P
                "Poor": "P", "P": "P", "PR": "P",
                # U
                "U": "U", "u": "U", "Unknown": "U", "UNKNOWN": "U"
            },
            "fluorescence": {
                # None
                "NON": "None", "None": "None", "No": "None", "FL0": "None", "N": "None", "NEG": "None", "Negligible": "None",
                # Faint
                "Faint": "Faint", "FNT": "Faint", "FA": "Faint", "FL1": "Faint", "F": "Faint",
                # Medium
                "Medium": "Medium", "MED": "Medium", "FL2": "Medium", "M": "Medium",
                # Strong
                "Strong": "Strong", "STG": "Strong", "ST": "Strong", "FL3": "Strong", "S": "Strong", "STR": "Strong",
                # Very Strong
                "Very Strong": "Very Strong", "VST": "Very Strong", "FL4": "Very Strong", "VSTG": "Very Strong", "VSTR": "Very Strong", "VSTRONG": "Very Strong"
            },
            "clarity": {
                # FL
                "FL": "FL",
                # IF
                "IF": "IF",
                # VVS1
                "VVS1": "VVS1", "VVS 1": "VVS1",
                # VVS2
                "VVS2": "VVS2", "VVS 2": "VVS2",
                # VS1
                "VS1": "VS1", "VS 1": "VS1", "VS1+": "VS1",
                # VS2
                "VS2": "VS2", "VS 2": "VS2", "VS2+": "VS2",
                # SI1
                "SI1": "SI1", "SI 1": "SI1", "SI1+": "SI1",
                # SI2
                "SI2": "SI2", "SI 2": "SI2",
                # SI3
                "SI3": "SI3", "SI 3": "SI3",
                # I1
                "I1": "I1", "I 1": "I1",
                # I2
                "I2": "I2", "I 2": "I2",
                # I3
                "I3": "I3", "i3": "I3", "I 3": "I3",
                # U (Unknown)
                "U": "U", "Unknown": "U", "UNKNOWN": "U", "u": "U"
            },
            "culet": {
                "None": "None", "Very Small": "VS", "Small": "S", "Slightly Large": " SL",
                "Medium": "M", "Large": "L", "Extra Large": "EL", "Very Large": "VL"
            },
            "color": {
                # D
                "D": "D", "d": "D", "D+": "D",
                # E
                "E": "E", "e": "E", "E+": "E",
                # F
                "F": "F", "f": "F", "F+": "F",
                # G
                "G": "G", "g": "G", "G+": "G",
                # H
                "H": "H", "H+": "H",
                # I
                "I": "I", "i": "I", "I+": "I",
                # J
                "J": "J", "j": "J", "J+": "J",
                # K
                "K": "K", "k": "K", "K+": "K",
                # L
                "L": "L", "l": "L", "L+": "L",
                # M
                "M": "M", "m": "M", "M+": "M",
                # N
                "N": "N", "n": "N", "N+": "N",
                # OP
                "O-P": "OP", "o-p": "OP", "O": "OP", "p": "OP", "OP": "OP", "op": "OP",
                # QR
                "Q-R": "QR", "q-r": "QR", "q": "QR", "r": "QR", "QR": "QR", "qr": "QR",
                # ST
                "S-T": "ST", "s-t": "ST", "s": "ST", "t": "ST", "ST": "ST", "st": "ST",
                # UV
                "U-V": "UV", "u-v": "UV", "u": "UV", "v": "UV", "UV": "UV", "uv": "UV",
                # WX
                "W-X": "WX", "w-x": "WX", "w": "WX", "x": "WX", "WX": "WX", "wx": "WX",
                # YZ
                "Y-Z": "YZ", "y-z": "YZ", "y": "YZ", "z": "YZ", "YZ": "YZ", "yz": "YZ"
            },
            "eye clean": {
                "YES": "YES", "NO": "NO"
            },
            "heart and arrow": {
                "YES": "YES", "NO": "NO"
            },
            "lab": {
                # IOD
                "IOD": "IOD", "iod": "IOD",
                # GIA
                "GIA": "GIA",
                # Other
                "Other": "Other",
                # None
                "NONE": "None", "none": "None",
                # HRD, IGI, IOD India (from previous mapping)
                "HRD": "HRD", "IGI": "IGI", "IOD India": "IOD India"
            },
            "shape": {
                # Round
                "BR": "Round", "ROUND": "Round", "RBC": "Round", "Round": "Round", "Round Brilliant": "Round", "round": "Round", "rbc": "Round", "rd": "Round", "RD": "Round", "RO": "Round", "ROUND BRILLIANT": "Round",
                # Oval
                "O": "Oval", "OV": "Oval", "OMB": "Oval", "OB": "Oval", "OVAL": "Oval", "Oval": "Oval", "omb": "Oval", "Omb": "Oval", "o": "Oval", "ov": "Oval", "OVL": "Oval", "Oval Brilliant": "Oval",
                # Pear
                "P": "Pear", "PS": "Pear", "PSH": "Pear", "PB": "Pear", "PMB": "Pear", "PEAR": "Pear", "Pear": "Pear", "pb": "Pear", "ps": "Pear", "p": "Pear", "psh": "Pear", "pmb": "Pear", "PR": "Pear",
                # Emerald
                "E": "Emerald", "EM": "Emerald", "EC": "Emerald", "SX": "Emerald", "Emerald": "Emerald", "EMERALD": "Emerald", "em": "Emerald", "Em": "Emerald", "ec": "Emerald",
                # Heart
                "H": "Heart", "HS": "Heart", "HT": "Heart", "MHRC": "Heart", "HB": "Heart", "HEART": "Heart", "Heart": "Heart", "hb": "Heart", "ht": "Heart", "hs": "Heart", "h": "Heart",
                # Marquise
                "MQB": "Marquise", "M": "Marquise", "MQ": "Marquise", "MB": "Marquise", "MARQUISE": "Marquise", "Marquise": "Marquise", "mqb": "Marquise", "mq": "Marquise", "m": "Marquise",
                # Cushion
                "CB": "Cushion", "Cushion Brilliant": "Cushion", "C": "Cushion", "CUX": "Cushion", "CU": "Cushion", "CMB": "Cushion", "CUSH": "Cushion", "CUS": "Cushion", "RCRMB": "Cushion", "CRC": "Cushion", "CSC": "Cushion", "CX": "Cushion", "RCSB": "Cushion", "SCMB": "Cushion", "SCX": "Cushion", "Cushion": "Cushion", "cushion": "Cushion", "Cushion Modified Brilliant": "Cushion", "CUMBR": "Cushion", "CUSHION MODIFIED": "Cushion", "LR_BRILLIANT": "Cushion", "CS": "Cushion", "CM": "Cushion", "CUSHION": "Cushion",
                # Radiant
                "R": "Radiant", "RAD": "Radiant", "RA": "Radiant", "RC": "Radiant", "RDN": "Radiant", "CRB": "Radiant", "RCRB": "Radiant", "Sq Radiant": "Radiant", "SQR": "Radiant", "CCSMB": "Radiant", "RADIANT": "Radiant", "Radiant": "Radiant", "Square Radiant": "Radiant", "LONG RADIANT": "Radiant", "RN": "Radiant",
                # Princess
                "PRN": "Princess", "PR": "Princess", "PRIN": "Princess", "PN": "Princess", "MDSQB": "Princess", "SMB": "Princess", "PRINCESS": "Princess", "Princess": "Princess", "smb": "Princess", "PC": "Princess",
                # Square Emerald
                "A": "Square Emerald", "CSS": "Square Emerald", "CSSC": "Square Emerald", "AC": "Square Emerald", "SE": "Square Emerald", "Asscher": "Square Emerald", "ASSCHER": "Square Emerald", "SQUARE EMERALD": "Square Emerald", "Square Emerald": "Square Emerald", "Square emerald": "Square Emerald", "SQ Emerald": "Square Emerald", "sq emerald": "Square Emerald", "Asscher Cut": "Square Emerald", "SQEM": "Square Emerald", "SQE": "Square Emerald", "ASSCHER CUT": "Square Emerald", "Square Emerald Cut": "Square Emerald", "SQUARE EMERALD CUT": "Square Emerald",
                # Baguette
                "Baguette": "Baguette", "BAG": "Baguette", "BG": "Baguette", "BAGUETTE": "Baguette", "Bag": "Baguette", "TAPERED BAGUETTE": "Baguette", "BAGETTE": "Baguette",
                # Taper
                "Taper": "Taper", "TAPER": "Taper", "taper": "Taper",
                # Rose
                "rose": "Rose", "RS": "Rose", "RRC": "Rose", "Rose": "Rose", "ROSE": "Rose", "ROSE-CUT": "Rose",
                # Shield
                "Shield": "Shield", "SHD": "Shield", "SHIELD": "Shield",
                # Trilliant
                "TR": "Trilliant", "TRI": "Trilliant", "Trill": "Trilliant", "Trilliant": "Trilliant", "TRILLIANT": "Trilliant",
                # Other
                "X": "Other", "BAT": "Other", "Starlet": "Other", "STARLET": "Other"
            }}
        normalized = data.copy()
        for field, mapping in mappings.items():
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
    
    

    async def find_diamonds(self, user_query: str, chat_history: list = None, session_id: str = None) -> Dict[str, Any]:
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
        


        # Prepare summary prompt
        prompt_2 = PromptTemplate(
            input_variables=["diamonds", "user_query", "total_count"],
            template="""
            You are an expert gemologist and diamond consultant.

            The user asked: "{user_query}"

            You have retrieved the following diamonds from the database:

            {diamonds}

            Now do the following:

            1. Begin with this line exactly:
            We found total {total_count} stones based on your query.

            2. Then say:
            Here we are displaying top 10 stones:

            3. Print each diamond on a **new line**, following this format:
            . Shape: <Shape>, Color: <Color>, Cut: <Cut>, Clarity: <Clarity>, Polish: <Polish>, Weight: <Carat>, Price/Carat: <$Price>

            4. Ensure that each diamond is printed on a separate line with a line break (`\n`).

            5. After listing the diamonds, provide a **2-3 sentence summary** of what makes this selection valuable and how it fits the user’s preferences.

            Make sure the response preserves the line breaks exactly as instructed, so it renders properly in a web frontend.
            """
        )
        summary_prompt = prompt_2.format(
            diamonds=format_diamond_list(diamonds),
            user_query=user_query,
            total_count=total_count
        )

        # Build chat history for memory (if any)
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
        messages.append(HumanMessage(content=summary_prompt))

        # Run summary LLM with memory (multi-turn)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.llm.invoke, messages)
        messages.append(AIMessage(content=response.content))

        # Convert messages back to serializable chat history
        updated_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                updated_history.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                updated_history.append({'role': 'assistant', 'content': msg.content})

        return {
            "diamonds": diamonds,
            "summary": response.content,
            "sql_query": sql,
            "total_count": total_count
        }