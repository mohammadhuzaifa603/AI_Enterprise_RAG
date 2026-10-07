"""
Initialize the database (creates all tables). Safe to run repeatedly.

Usage:
    python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.database import init_db


def main():
    print(f"Initializing database at: {settings.database_url}")
    init_db()
    print("Done. Tables created (or already existed).")


if __name__ == "__main__":
    main()
