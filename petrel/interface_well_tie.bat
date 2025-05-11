@echo off
setlocal

echo Logs file: %1
echo Seismic file: %2
echo Well path file: %3
echo Table file: %4

set CONDA_ENV_ZIP=dlseis_well_tie.zip
set CONDA_ENV_NAME=wtie

set PATH_CONDA=D:\anaconda3

if not exist "%AppData%\teste.py" (
    echo Descompactando script python
    "%~dp0\7z.exe" x "%~dp0\%CONDA_ENV_ZIP" -o"%AppData%"
)

call %PATH_CONDA%\Scripts\activate.bat

if not exist "%PATH_CONDA%\envs\%CONDA_ENV_NAME%" (
    echo no env
    call conda env create -f "%CAMINHO%\dlseis_well_tie\enviroment_windows.yml"
)

call conda activate "%CONDA_ENV_NAME%"

:: Set the QT_PLUGIN_PATH environment variable
set QT_PLUGIN_PATH=%PATH_CONDA%\envs\%CONDA_ENV_NAME%\Library\plugins

:: this will have to be changed, however i don't currently know how to dinamycally change disks on windows

D:
cd D:\Caio\dlseis_well_tie_petrel\dlseis_well_tie_petrel\bin\Debug
:: python "%AppData%\dlseis_well_tie\petrel\teste.py" %1 %2 %3 %4