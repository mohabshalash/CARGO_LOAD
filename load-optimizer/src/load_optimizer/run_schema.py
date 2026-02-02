import os
from psycopg2 import connect
from dotenv import load_dotenv

load_dotenv()

conn = connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

with open("schema.sql", "r") as f:
    cur.execute(f.read())

conn.commit()
cur.close()
conn.close()

print("Schema applied successfully")