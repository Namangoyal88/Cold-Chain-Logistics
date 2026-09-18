import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.rag import build_vectorstore


if __name__ == "__main__":
    print("COLD-CHAIN SOP RAG INGESTION")
    build_vectorstore()
    print("RAG INGESTION COMPLETE")