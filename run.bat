@echo off
REM Convenience script: sets up a venv, installs deps, initializes the DB,
REM and prints the two commands you need to run the app.

cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

if not exist ".env" (
    echo Creating .env from .env.example...
    copy .env.example .env
)

echo Initializing database...
python scripts\init_db.py

echo.
echo Setup complete. In two separate terminals, run:
echo.
echo   .venv\Scripts\activate ^&^& uvicorn app.main:app --reload
echo   .venv\Scripts\activate ^&^& streamlit run frontend\streamlit_app.py
echo.
echo Then open http://localhost:8501 (frontend) and http://localhost:8000/docs (API docs).
