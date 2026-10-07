#!/usr/bin/env bash
# Convenience script: sets up a venv, installs deps, initializes the DB,
# and prints the two commands you need to run the app.
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

echo "Initializing database..."
python scripts/init_db.py

echo ""
echo "Setup complete. In two separate terminals, run:"
echo ""
echo "  source .venv/bin/activate && uvicorn app.main:app --reload"
echo "  source .venv/bin/activate && streamlit run frontend/streamlit_app.py"
echo ""
echo "Then open http://localhost:8501 (frontend) and http://localhost:8000/docs (API docs)."
