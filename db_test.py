from sqlalchemy import create_engine, inspect, text

engine = create_engine("postgresql+psycopg2://postgres:0207@localhost:5432/postgres")
inspector = inspect(engine)

# List tables
print("[TABLES]", inspector.get_table_names())

# List columns
print("[COLUMNS]", inspector.get_columns('diamonds'))

# Check row count
with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM diamonds")).fetchone()[0]
    print("[ROW COUNT]", count)

# Sample rows
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM diamonds LIMIT 5")).fetchall()
    print("[SAMPLE ROWS]", result)
