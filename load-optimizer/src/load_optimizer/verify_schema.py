import os
from dotenv import load_dotenv
from psycopg2 import connect

load_dotenv()

conn = connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema='public'
ORDER BY table_name;
""")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

# Check the 3 MVP tables exist
required = {"sessions", "messages", "session_state"}
missing = required - set(tables)
print("Missing:", missing)

cur.close()
conn.close()