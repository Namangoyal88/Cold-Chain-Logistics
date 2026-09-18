import sys
from pathlib import Path
from src.database import create_database


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    print("COLD-CHAIN DATABASE SETUP")
    create_database()
    print("DATABASE SETUP COMPLETE")