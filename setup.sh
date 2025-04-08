#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv command not found in PATH. Please install uv first."
    echo "See: https://github.com/astral-sh/uv"
    exit 1
fi
echo "Found uv."

# Get the absolute directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="$SCRIPT_DIR/.venv"

# --- Check if .venv exists and remove it ---
if [ -d "$VENV_DIR" ]; then
    echo "WARNING: Existing directory $VENV_DIR found."
    echo "         This script will remove it to create a fresh environment specifically for this project."
    echo "         Press Ctrl+C within 5 seconds to cancel if you need to back up this directory first."
    sleep 5
    echo "Removing $VENV_DIR..."
    rm -rf "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to remove existing $VENV_DIR."
        echo "       Please check permissions or ensure no processes are using it, then try again."
        exit 1
    fi
     echo "Existing .venv removed successfully."
fi
# --- End removal ---

echo "Creating virtual environment via 'uv venv'..."
# Run uv commands from the project directory to ensure context
# Explicitly tell uv where, relative to CWD (which is $SCRIPT_DIR here)
(cd "$SCRIPT_DIR" && uv venv .venv)

echo "Installing project dependencies via 'uv pip install -e .' ..."
(cd "$SCRIPT_DIR" && uv pip install -e .)

echo ""
echo "--- Setup Complete! ---"
echo ""
echo "To configure your MCP client (e.g., Claude Desktop), use the following settings:"
echo ""
echo "For the \"command\" field:"
echo "uv"
echo ""
echo "For the \"args\" field:"
# Ensure the path is quoted properly if it contains spaces, although JSON array handles this.
echo "[ \"run\", \"--cwd\", \"$SCRIPT_DIR\", \"mcp-function-generator\" ]"
echo ""
echo "Example JSON block for your client's config file:"
echo "  \"pythonFunctionGenerator\": {"
echo "    \"command\": \"uv\","
echo "    \"args\": [ \"run\", \"--cwd\", \"$SCRIPT_DIR\", \"mcp-function-generator\" ]"
echo "  }"
echo ""

exit 0