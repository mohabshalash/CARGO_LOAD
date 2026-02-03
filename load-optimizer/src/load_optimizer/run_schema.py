import os
from pathlib import Path
from dotenv import load_dotenv
from psycopg2 import connect

# repo root = two levels up from db/admin
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

print("Repo root:", REPO_ROOT)
print(".env path:", ENV_PATH, "exists:", ENV_PATH.exists())
print("schema.sql path:", SCHEMA_PATH, "exists:", SCHEMA_PATH.exists())

load_dotenv(ENV_PATH)

db_url = os.getenv("DATABASE_URL")
print("DATABASE_URL present:", bool(db_url))
if not db_url:
    raise SystemExit("DATABASE_URL not found. Check .env at repo root.")

sql_text = SCHEMA_PATH.read_text(encoding="utf-8")
print("SQL chars:", len(sql_text))
if len(sql_text) < 50:
    raise SystemExit("schema.sql is empty/too small. Paste your CREATE TABLE SQL into it.")

conn = connect(db_url)
cur = conn.cursor()

cur.execute("SELECT current_database(), current_user, inet_server_addr(), inet_server_port();")
print("Connected to:", cur.fetchone())

# Execute each statement safely
statements = [s.strip() for s in sql_text.split(";") if s.strip()]
print("Statements found:", len(statements))

for i, stmt in enumerate(statements, start=1):
    cur.execute(stmt + ";")
    # optional progress print:
    # print(f"Executed statement {i}/{len(statements)}")

conn.commit()
print("Committed.")

# Verify tables in SAME connection
cur.execute("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema='public'
ORDER BY table_name;
""")
tables = [r[0] for r in cur.fetchall()]
print("Tables now:", tables)

cur.close()
conn.close()
print("Done.")