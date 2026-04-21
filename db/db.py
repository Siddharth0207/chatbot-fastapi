import os
import re
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
EXCEL_PATH = BASE_DIR / "Diamond_Records_1000.xlsx"
DEFAULT_SUPABASE_URL = (
    "postgresql://postgres.qausdzscwkzfhhrmicof:[Juj7E0zO09wwQEFM]"
    "@aws-1-ap-northeast-1.pooler.supabase.com:6543/postgres"
)


def _normalize_postgres_url(raw_url: str) -> str:
    """Normalize URL style for SQLAlchemy sync engine + pandas.to_sql."""
    url = raw_url.strip().strip('"').strip("'")

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    if url.startswith("postgresql+asyncpg://"):
        url = "postgresql+psycopg2://" + url[len("postgresql+asyncpg://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://") :]

    # Supabase docs sometimes show passwords wrapped in [] placeholders.
    # Remove wrapper and URL-encode password safely.
    bracketed_pwd = re.search(r":\[([^\]]+)\]@", url)
    if bracketed_pwd:
        encoded_pwd = quote_plus(bracketed_pwd.group(1))
        url = re.sub(r":\[[^\]]+\]@", f":{encoded_pwd}@", url, count=1)

    return url


def main() -> None:
    database_url = _normalize_postgres_url(os.getenv("DATABASE_URL", DEFAULT_SUPABASE_URL))

    # Load Excel
    df = pd.read_excel(EXCEL_PATH)

    # Optional: Rename or clean columns if needed
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Ensure boolean fields are correct
    if "is_eye_clean" in df.columns:
        df["is_eye_clean"] = df["is_eye_clean"].astype(bool)
    if "is_heart_arrow" in df.columns:
        df["is_heart_arrow"] = df["is_heart_arrow"].astype(bool)

    # Setup SQLAlchemy engine
    engine = create_engine(database_url)

    # Push to PostgreSQL
    df.to_sql("diamond_db", engine, if_exists="append", index=False)

    print("Data pushed successfully.")


if __name__ == "__main__":
    main()
