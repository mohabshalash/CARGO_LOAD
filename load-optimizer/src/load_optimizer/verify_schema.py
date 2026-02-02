import os
from psycopg2 import connect
from dotenv import load_dotenv

load_dotenv()
conn = connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
""")

print(cur.fetchall())

cur.close()
conn.close()