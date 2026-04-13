import pandas as pd
from sqlalchemy import create_engine

# Load Excel
df = pd.read_excel(r"C:\Users\siddh\OneDrive\文档\cb\chatbot-fastapi\Diamond_Records_1000.xlsx")

# Optional: Rename or clean columns if needed
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Ensure boolean fields are correct
if "is_eye_clean" in df.columns:
    df["is_eye_clean"] = df["is_eye_clean"].astype(bool)
if "is_heart_arrow" in df.columns:
    df["is_heart_arrow"] = df["is_heart_arrow"].astype(bool)

# Setup SQLAlchemy engine for SQL Server
# Replace DRIVER, SERVER, DATABASE, UID, and PWD with your actual SQL Server details
engine = create_engine(
    "mysql+pymysql://root:0207@localhost:3306/discord"
)

# Push to SQL Server
df.to_sql("diamonds", engine, if_exists="append", index=False)

print("Data pushed to SQL Server successfully.")
