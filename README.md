# MCP Python Function Generator Server

An MCP server that exposes a tool to generate Python function code based on a description, with optional human-provided testing.

## Setup (Easy Install with uv)

This server is designed to be easy to set up using the `uv` Python package manager.

1.  **Prerequisites:** Ensure you have `uv` installed and available in your system's PATH. See [uv installation instructions](https://github.com/astral-sh/uv).
2.  **Get the Code:** Clone this repository or download and extract the source code ZIP file.
3.  **Run Setup Script:**
    *   **Windows:** Open Command Prompt or PowerShell, navigate (`cd`) to the project directory (e.g., `MCP_Server`), and run: `.\setup.bat`
    *   **Linux/macOS:** Open your terminal, navigate (`cd`) to the project directory, make the script executable (`chmod +x setup.sh`), and run: `./setup.sh`
    The script will use `uv` to create a local virtual environment (`.venv`) and install the necessary dependencies. It will also print the exact command and arguments needed for client configuration.
    *(Note: The script will attempt to remove any existing `.venv` directory in the project folder to ensure a clean environment. Back up its contents first if needed.)*

4.  **Configure Your MCP Client:** Copy the `command` ("uv") and `args` array (`[ "run", "--project", "/path/to/your/project", "mcp-function-generator" ]`) printed by the setup script into your MCP client's configuration file (e.g., Claude Desktop's `settings.json`).

    Example JSON block (replace the path with the one from your setup script output):
    ```json
    {
      "mcpServers": {
        "pythonFunctionGenerator": {
          "command": "uv",
          "args": [
            "run",
            "--project",
            // --- Paste ABSOLUTE path from setup script output here ---
            "/path/to/your/project/MCP_Server", // Example Linux/macOS
            // "C:\\Users\\YourName\\Projects\\MCP_Server", // Example Windows (escaped!)
            // --- End Path ---
            "mcp-function-generator"
           ]
        }
        // Add other servers like filesystem if needed
      }
    }
    ```
    *(Remember that backslashes `\` in Windows paths MUST be escaped as `\\` in JSON strings)*

5.  **Restart Client:** Save the configuration file and restart your MCP client (e.g., Claude Desktop) for the changes to take effect.

## Usage

Once configured in your client, the MCP server will be automatically started when needed.

The server exposes a single tool named `create_python_function`.

**Arguments:**

*   `requirement` (string, **required**):
    *   A natural language description of the function needed. Include examples, constraints, or edge cases if possible.
    *   Example: `"Create a function that takes a list of strings and returns the longest string. Handle empty lists."`

*   `test_cases` (array of objects, **optional**):
    *   A list of **human-written and validated** test cases to verify the generated code's functionality.
    *   **Structure:** `[ { "inputs": { "param_key1": value1, ... }, "expected": expected_value }, ... ]`
    *   **`inputs`**: A dictionary where keys are the desired parameter names for the function, and values are the inputs for that test case.
    *   **`expected`**: The single expected output value for those inputs. *(Note: Currently supports exact equality `==` checking only. Does not handle non-deterministic outputs or floating-point tolerances automatically.)*.
    *   **IMPORTANT Constraint:** If you provide `test_cases`, the keys used in the `inputs` dictionary of the *first* test case **will determine the exact parameter names** the generated function must use. Ensure keys are consistent across all your test cases.
    *   **!! WARNING !!**: This argument is intended **only** for tests written or validated by a human. **Do NOT ask an AI client (like Claude) to generate test cases for this argument**, as their correctness cannot be guaranteed by this tool and may lead to unexpected validation failures or incorrect code. If you want untested code generated quickly, simply omit the `test_cases` argument.

**Execution Flow:**

1.  The server receives the `requirement`.
2.  It generates a function specification (name, parameters, return type, docstring).
    *   If `test_cases` were provided, the generated parameter names **must** match the keys from the `inputs` dictionaries (based on the first test case). The server validates this match.
3.  It generates the Python code based on the specification.
4.  It validates the code for basic syntax and function definition.
5.  If `test_cases` were provided *and* the spec generation used the correct parameter keys, it executes the code against each test case using keyword arguments.
6.  It returns the result, including the spec, the code, and test results (if run and successful). If tests fail after retries, or if spec generation fails validation against test keys, an error is returned.

## Example Client Calls (Conceptual JSON)

**1. Generate code without tests:**

```json
{
  "tool_name": "create_python_function",
  "arguments": {
    "requirement": "Write a function `add_one` that takes an integer `x` and returns `x + 1`."
  }
}