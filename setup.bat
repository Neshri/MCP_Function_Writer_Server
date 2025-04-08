@echo off
setlocal enabledelayedexpansion

echo --- Script Start ---
echo Checking for uv...
uv --version > nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: uv command not found in PATH. Please install uv first.
    echo See: https://github.com/astral-sh/uv
    GOTO EndScriptWithError_NoUV
)
echo Found uv.

REM --- Define Paths ---
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "VENV_DIR=.venv"
set "VENV_PATH=%PROJECT_DIR%\%VENV_DIR%"
echo DEBUG: Project Dir = [%PROJECT_DIR%]
echo DEBUG: Venv Path   = [%VENV_PATH%]

REM --- Check Existence and Branch ---
echo DEBUG: Checking for existing venv: "%VENV_PATH%\"
if exist "%VENV_PATH%\" (
    echo INFO: Existing venv directory found.
    GOTO DeleteVenv
) ELSE (
    echo INFO: No existing venv directory found.
    GOTO CreateVenv
)
REM --- Should not reach here ---
echo ERROR: Logic flow error after existence check.
GOTO EndScriptWithError


:DeleteVenv
    echo DEBUG: Entered :DeleteVenv block.
    echo ===============================================================================
    echo WARNING: Existing directory "%VENV_PATH%" found.
    echo          This script needs to remove it to create a fresh environment.
    echo          (Ensure no apps are locking files inside this directory!)
    echo ===============================================================================
    echo.
    echo Press Ctrl+C within 5 seconds to cancel...
    timeout /t 5 /nobreak > nul

    echo Attempting to remove "%VENV_PATH%"...
    rmdir /s /q "%VENV_PATH%"
    set RMDIR_ERRORLEVEL=!errorlevel!
    echo DEBUG: rmdir command finished. Errorlevel = !RMDIR_ERRORLEVEL!

    if !RMDIR_ERRORLEVEL! neq 0 (
        echo ERROR: Failed to remove existing "%VENV_PATH%". Errorlevel: !RMDIR_ERRORLEVEL!.
        if !RMDIR_ERRORLEVEL! equ 5 echo        (Hint: Error 5 usually means 'Access is denied' - check file locks/permissions)
        echo        Please try manually deleting the "%VENV_PATH%" directory, then re-run this script.
        GOTO EndScriptWithError
    )

    REM Double check it's actually gone after supposed success
    if exist "%VENV_PATH%\" (
       echo ERROR: rmdir reported success BUT "%VENV_PATH%" directory still exists! Filesystem issue?
       GOTO EndScriptWithError
    )

    echo INFO: Removal of "%VENV_PATH%" succeeded.
    echo DEBUG: Jumping from :DeleteVenv to :CreateVenv
    GOTO CreateVenv


:CreateVenv
    echo DEBUG: Entered :CreateVenv block.
    echo INFO: Proceeding to virtual environment creation...
    pushd "%PROJECT_DIR%"
    if !errorlevel! neq 0 (
        echo ERROR: Failed to change directory to "%PROJECT_DIR%" before uv venv.
        GOTO EndScriptWithError
    )
    echo DEBUG: Changed directory to %CD%. Running 'uv venv "%VENV_DIR%"'...
    uv venv "%VENV_DIR%"
    set UV_VENV_ERRORLEVEL=!errorlevel!
    echo DEBUG: 'uv venv' finished. Errorlevel = !UV_VENV_ERRORLEVEL!
    popd
    if !UV_VENV_ERRORLEVEL! neq 0 (
        echo ERROR: Failed to create virtual environment using 'uv venv'. Errorlevel: !UV_VENV_ERRORLEVEL!.
        GOTO EndScriptWithError
    )
    echo INFO: Virtual environment created successfully.
    echo DEBUG: Jumping from :CreateVenv to :InstallDeps
    GOTO InstallDeps


:InstallDeps
    echo DEBUG: Entered :InstallDeps block.
    echo INFO: Installing project dependencies...
    pushd "%PROJECT_DIR%"
     if !errorlevel! neq 0 (
        echo ERROR: Failed to change directory to "%PROJECT_DIR%" before uv pip install.
        GOTO EndScriptWithError
    )
    echo DEBUG: Changed directory to %CD%. Running 'uv pip install -e .'
    uv pip install -e .
    set UV_INSTALL_ERRORLEVEL=!errorlevel!
    echo DEBUG: 'uv pip install' finished. Errorlevel = !UV_INSTALL_ERRORLEVEL!
    popd
    if !UV_INSTALL_ERRORLEVEL! neq 0 (
        echo ERROR: Failed to install dependencies using 'uv pip install'. Errorlevel: !UV_INSTALL_ERRORLEVEL!.
        GOTO EndScriptWithError
    )
    echo INFO: Dependencies installed successfully.
    echo DEBUG: Jumping from :InstallDeps to :Success
    GOTO Success


:Success
    echo DEBUG: Entered :Success block.
    echo.
    echo --- Setup Complete! ---
    echo.
    echo To configure your MCP client (e.g., Claude Desktop), use the following settings:
    echo.
    echo For the "command" field:
    echo uv
    echo.
    echo For the "args" field:
    echo [ "run", "--project", "%PROJECT_DIR%", "mcp-function-generator" ]
    echo.
    echo Example JSON block for your client's config file:
    echo   "pythonFunctionGenerator": {
    echo     "command": "uv",
    echo     "args": [ "run", "--project", "%PROJECT_DIR%", "mcp-function-generator" ]
    echo   }
    echo.
    echo (Note: Backslashes in the --project path might need to be escaped like \\ in JSON)
    echo.
    GOTO EndScript


:EndScriptWithError_NoUV
    REM Separate exit point if uv wasn't found initially
    exit /b 1

:EndScriptWithError
    echo --- Setup Failed ---
    endlocal
    exit /b 1

:EndScript
    echo --- Script End ---
    endlocal
    exit /b 0