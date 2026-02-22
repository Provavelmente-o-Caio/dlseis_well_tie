@echo off
setlocal EnableDelayedExpansion

set ENV_ZIP=dlseis_well_tie.zip
set EXTRACT_PATH=%AppData%
set PROJECT_PATH=%EXTRACT_PATH%\dlseis_well_tie
set VENV_PATH=%PROJECT_PATH%\.venv
set SCRIPT_PATH=%PROJECT_PATH%\petrel\basic_well_path.py
set REQUIREMENTS=%PROJECT_PATH%\requirements.txt
set WRAPPER_PATH=%~dp0run_isolated.py

if not exist "%SCRIPT_PATH%" (
    echo [INFO] Descompactando ambiente...
    "%~dp07z.exe" x "%~dp0%ENV_ZIP%" -o"%EXTRACT_PATH%"
)

if not exist "%VENV_PATH%" (
    echo [INFO] Criando ambiente virtual...
    python -m venv "%VENV_PATH%"
    
    echo [INFO] Instalando dependencias...
    call "%VENV_PATH%\Scripts\activate.bat"
    python -m pip install --upgrade pip
    pip install -r "%REQUIREMENTS%"
) else (
    echo [INFO] Ativando ambiente virtual...
    call "%VENV_PATH%\Scripts\activate.bat"
)

echo.
echo ========================================
echo DIAGNOSTICO PRE-EXECUCAO
echo ========================================
python --version
pip --version
where hdf5.dll 2>nul
echo ========================================
echo.

echo [INFO] Executando script via wrapper isolado...
python "%WRAPPER_PATH%" "%SCRIPT_PATH%" %*

set EXIT_CODE=!ERRORLEVEL!

echo.
if !EXIT_CODE! equ 0 (
    echo [SUCESSO] Script executado com sucesso
) else (
    echo [ERRO] Script terminou com codigo: !EXIT_CODE!
)

call deactivate
endlocal
exit /b !EXIT_CODE!