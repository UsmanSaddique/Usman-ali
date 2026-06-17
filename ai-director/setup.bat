@echo off
echo ==============================================
echo AI Director - Environment Setup
echo ==============================================

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo ==============================================
echo Setup Complete!
echo You can now start the server by running:
echo     venv\Scripts\activate
echo     python run.py
echo ==============================================
pause
