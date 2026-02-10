@echo off
setlocal EnableDelayedExpansion

set CONDA_ENV_ZIP=dlseis_well_tie.zip
set PATH_CONDA=D:\anaconda3
set EXTRACT_PATH=%AppData%
set SCRIPT_PATH=%EXTRACT_PATH%\dlseis_well_tie\petrel\basic_well_path.py
set WRAPPER_PATH=%~dp0run_isolated.py

if not exist "%SCRIPT_PATH%" (
    echo [INFO] Descompactando ambiente...
    "%~dp07z.exe" x "%~dp0%CONDA_ENV_ZIP%" -o"%EXTRACT_PATH%"
)

echo [INFO] Ativando Conda...
call "%PATH_CONDA%\Scripts\activate.bat"

if not exist "%PATH_CONDA%\envs\wtie" (
    echo [INFO] Criando ambiente wtie...
    call conda env create -f "%EXTRACT_PATH%\dlseis_well_tie\enviroment_windows.yml"
)

echo [INFO] Ativando ambiente wtie...
call conda activate wtie

echo.
echo ========================================
echo DIAGNOSTICO PRE-EXECUCAO
echo ========================================
python --version
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

call conda deactivate
endlocal
exit /b !EXIT_CODE!