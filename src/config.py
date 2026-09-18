from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
POLICY_DIR = DATA_DIR / "policy"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"


PROCESSED_DATA_DIR.mkdir(parents = True, exist_ok = True)
VECTORSTORE_DIR.mkdir(parents = True, exist_ok = True)


CSV_FILE = RAW_DATA_DIR / "dynamic_supply_chain_logistics_dataset.csv"

DATABASE_FILE = PROCESSED_DATA_DIR / "fleet.db"

SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"


load_dotenv(PROJECT_ROOT / ".env")


AGENT_LLM = os.getenv("AGENT_LLM", "OLLAMA").strip().upper()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


OPENAI_MODEL = os.getenv( "OPENAI_MODEL", "gpt-4o-mini")
DEEPSEEK_MODEL = os.getenv( "DEEPSEEK_MODEL", "deepseek-chat")
OLLAMA_MODEL = os.getenv( "OLLAMA_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.getenv( "EMBEDDING_MODEL", "BAAI/bge-m3")


WEATHER_API_TIMEOUT = int(os.getenv("WEATHER_API_TIMEOUT", "10"))
MAX_DATABASE_ROWS = int(os.getenv("MAX_DATABASE_ROWS", "20"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))