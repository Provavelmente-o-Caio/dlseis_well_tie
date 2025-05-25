@echo off
setlocal

echo Logs file: %1
echo Seismic file: %2
echo Well path file: %3
echo Table file: %4

set CONDA_ENV_ZIP=dlseis_well_tie.zip
set CONDA_ENV_NAME=wtie

set PATH_CONDA=D:\anaconda3
set EXTRACT_PATH=%AppData%


if not exist "%AppData%\dlseis_well_tie\petrel\interface2.py" (
	echo Descompactando script python
	"%~dp07z.exe" x "%~dp0%CONDA_ENV_ZIP%" -o%EXTRACT_PATH%
	pause
)

call %PATH_CONDA%\Scripts\activate.bat

if not exist "%PATH_CONDA%\envs\%CONDA_ENV_NAME%" (
    echo no env
    call conda env create -f "%AppData%\dlseis_well_tie\enviroment_windows.yml"
)

echo activating env
call conda activate "%CONDA_ENV_NAME%"

:: Set the QT_PLUGIN_PATH environment variable
set QT_PLUGIN_PATH=%PATH_CONDA%\envs\%CONDA_ENV_NAME%\Library\plugins

:: this will have to be changed, however i don't currently know how to dinamycally change disks on windows

echo executing code
python "%AppData%\dlseis_well_tie\petrel\interface2.py" %1 %2 %3 %4 || (
    echo Erro ao executar o script Python
    pause
)
pause