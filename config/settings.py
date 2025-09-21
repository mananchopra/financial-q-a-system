"""Configuration settings for the Financial Q&A System."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables")

# Model Configuration
LLM_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"
COLLECTION_NAME = "financial_filings"

# Chunking Configuration
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
MAX_CHUNKS_PER_QUERY = 8

# Company Information
COMPANIES = {
    "GOOGL": {"name": "Google", "cik": "1652044"},
    "MSFT": {"name": "Microsoft", "cik": "789019"}, 
    "NVDA": {"name": "NVIDIA", "cik": "1045810"}
}

# SEC Configuration
SEC_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_HEADERS = {
    "User-Agent": "Financial Q&A System (your-email@example.com)"
}

# Years to process
YEARS = [2022, 2023, 2024]

# File paths
DATA_DIR = "./data/filings"
