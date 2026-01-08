python --version >nul 2>&1

if not exist "venv" (
    python -m venv venv
)

call venv\Scripts\activate

pip install -r requirements.txt

python cnnTorch.py

cmd /k