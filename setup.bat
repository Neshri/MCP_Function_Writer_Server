@echo off
setlocal enabledelayedexpansion

echo Checking for uv...
uv --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: uv command not found in PATH. Please install uv first.
    exit /b 1
)
echo Found uv.

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "VENV_DIR=.venv"
set "VENV_PATH=%PROJECT_DIR%\%VENV_DIR%"

echo DEBUG: Project Dir = [%PROJECT_DIR%]
echo DEBUG: Venv Path   = [%VENV_PATH%]
echo DEBUG: Checking filesystem for directory: "%VENV_PATH%\"

REM --- Check if directory exists ---
if exist "%VENV_PATH%\" (
    echo DEBUG: Directory "%VENV_PATH%" found. Will attempt deletion.
    GOTO DeleteVenv
) ELSE (
    echo DEBUG: Directory "%VENV_PATH%" not found. Skipping deletion.
    GOTO CreateVenv
)


:DeleteVenv
    echo INFO: Existing directory "%VENV_PATH%" detected. Attempting removal.
    echo       (Ensure no apps are locking files inside!)
    echo       Press Ctrl+C within 5 seconds to cancel...
    timeout /t 5 /nobreak > nul

    echo Attempting: rmdir /s /q "%VENV_PATH%"
    rmdir /s /q "%VENV_PATH%"
    set RMDIR_ERRORLEVEL=%errorlevel%
    echo DEBUG: rmdir command finished. Errorlevel = !RMDIR_ERRORLEVEL!

    if !RMDIR_ERRORLEVEL! neq 0 (
        echo ERROR: Failed to remove existing "%VENV_PATH%". Errorlevel: !RMDIR_ERRORLEVEL!.
        if !RMDIR_ERRORLEVEL! equ 2 echo        (Error 2 likely means 'System cannot find the file specified' - should not happen here!)
        if !RMDIR_ERRORLEVEL! equ 5 echo        (Error 5 likely means 'Access is denied' - check file locks/permissions)
        echo        Please try manually deleting the "%VENV_PATH%" directory, then re-run this script.
        GOTO EndScriptWithError
    )

    REM Double check it's actually gone after supposed success
    if exist "%VENV_PATH%\" (
       echo ERROR: rmdir reported success BUT "%VENV_PATH%" directory still exists! Filesystem issue?
       GOTO EndScriptWithError
    )
    echo INFO: Removal of "%VENV_PATH%" succeeded.
    GOTO CreateVenv


:CreateVenv
    echo INFO: Proceeding to virtual environment creation.
    echo Creating virtual environment via 'uv venv'...
    pushd "%PROJECT_DIR%"
    uv venv "%VENV_DIR%"
    set UV_VENV_ERRORLEVEL=%errorlevel%
    popd
    if !UV_VENV_ERRORLEVEL! neq 0 (
        echo ERROR: Failed to create virtual environment using 'uv venv'. Errorlevel: !UV_VENV_ERRORLEVEL!.
        GOTO EndScriptWithError
    )
    GOTO InstallDeps


:InstallDeps
    echo Installing project dependencies via 'uv pip install -e .' ...
    pushd "%PROJECT_DIR%"
    uv pip install -e .
    set UV_INSTALL_ERRORLEVEL=%errorlevel%
    popd
    if !UV_INSTALL_ERRORLEVEL! neq 0 (
        echo ERROR: Failed to install dependencies using 'uv pip install'. Errorlevel: !UV_INSTALL_ERRORLEVEL!.
        GOTO EndScriptWithError
    )
    GOTO Success


:Success
    echo.
    echo --- Setup Complete! ---
    echo.
    echo To configure your MCP client (e.g., Claude Desktop), use the following settings:
    echo.
    echo For the "command" field:
    echo uv
    echo.
    echo For the "args" field:
    echo [ "run", "--cwd", "%PROJECT_DIR%", "mcp-function-generator" ]
    echo.
    echo Example JSON block for your client's config file:
    echo   "pythonFunctionGenerator": {
    echo     "command": "uv",
    echo     "args": [ "run", "--cwd", "%PROJECT_DIR%", "mcp-function-generator" ]
    echo   }
    echo.
    echo (Note: Backslashes in the --cwd path might need to be escaped like \\ in JSON)
    echo.
    GOTO EndScript


:EndScriptWithError
    echo Setup failed.
    exit /b 1

:EndScript
    endlocal
    exit /b 0