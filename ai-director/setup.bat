@echo off
echo ==============================================
echo AI Director - Environment Setup (Using ComfyUI)
echo ==============================================

set COMFY_PYTHON=C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\python_embeded\python.exe

echo Installing llama-cpp-python (CPU wheel)...
"%COMFY_PYTHON%" -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

echo Installing remaining dependencies into ComfyUI Python environment...
"%COMFY_PYTHON%" -m pip install -r requirements.txt

echo ==============================================
echo Setup Complete!
echo You can now start the server by running:
echo     run_server.bat
echo ==============================================
pause
