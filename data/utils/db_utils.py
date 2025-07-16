import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_pg_engine():
    """
    Create and return a PostgreSQL engine with connection pooling.
    Returns:
        SQLAlchemy engine
    """
    load_dotenv()
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT")
    db = os.getenv("PG_DB")
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}",
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )
        logger.info("PostgreSQL engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL engine: {str(e)}")
        raise