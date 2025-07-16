import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_pg_engine():
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")
    db = os.getenv("PG_DB")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}") 