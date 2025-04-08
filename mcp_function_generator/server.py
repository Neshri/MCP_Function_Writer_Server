# mcp_function_generator/server.py
import logging
import anyio
import click
import mcp.types as types
from mcp.server.lowlevel import Server
import json # For pretty printing schema in description
from .ai_program_writer import AIProgramWriter, AISettings, AIWriterError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AIProgramWriter")
settings = AISettings() # type: ignore
ai_writer = AIProgramWriter(settings)


# Placeholder function for generating the python code
# In a real scenario, this might call an LLM API or use more sophisticated templating.
def generate_python_function_code(
    function_name: str,
    arguments_description: str,
    return_description: str,
    logic_description: str,
) -> str:
    """
    Generates a simple Python function string based on descriptions.
    This is a placeholder implementation.
    """
    # Basic attempt to parse args - very naive, real implementation needs more robustness or LLM help
    args_list_str = f"# Args based on: {arguments_description}"

    code = f"""\
import typing

def {function_name}({args_list_str}) -> {return_description}:
    \"\"\"
    {logic_description}

    Args:
        {arguments_description} # Consider refining this part based on structured input if possible

    Returns:
        {return_description}
    \"\"\"
    # --- BEGIN Placeholder Logic ---
    # Implement the logic described: {logic_description}
    # Example placeholder based on return type hint:
    print(f"Function '{function_name}' called!")
    print(f"Arguments description: {arguments_description}")
    print(f"Logic to implement: {logic_description}")

    # Replace this with actual implementation based on logic_description
    pass # Placeholder implementation
    # --- END Placeholder Logic ---

# Example Usage (optional, for illustration)
# result = {function_name}(...)
# print(result)

"""
    return code


# Define the input schema for the tool
# Following JSON Schema structure
CREATE_FUNCTION_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "requirement": {
            "type": "string",
            "description": "Natural language description of the function needed, including examples or constraints.",
        },
        "test_cases": {
            "type": "array",
            "description": "(Optional) A list of **human-written and validated** test cases. The AI calling this tool should NOT generate these itself. Keys in the 'inputs' dict dictate parameter names.", # Added emphasis and warning
            "items": {
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "object",
                        "description": "Dictionary of input arguments (keys MUST match expected function parameters).", # Added key matching note
                        "additionalProperties": True
                    },
                    "expected": {
                        "description": "The single expected output value for the given inputs."
                    }
                },
                "required": ["inputs", "expected"]
            },
            "minItems": 1
        }
    },
    "required": ["requirement"],
}


@click.command()
@click.option("--port", default=8001, help="Port to listen on for SSE") # Changed default port
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    # Changed server name
    app = Server("mcp-function-generator")

    @app.call_tool()
    async def create_function_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent]:
        tool_name = "create_python_function"
        # ... (tool name check) ...

        if "requirement" not in arguments:
            raise ValueError("Missing required argument 'requirement'")

        requirement_text = arguments["requirement"]
        # --- Get optional test cases ---
        human_tests_data = arguments.get("test_cases") # Returns None if not present

        # Instantiate writer
        # settings = AISettings() # Assuming settings are loaded globally or passed
        # ai_writer = AIProgramWriter(settings) # Assuming writer is available

        logger.info(f"Received request for tool '{tool_name}'. Tests provided: {human_tests_data is not None}")

        try:
            # --- Pass test data (or None) to the writer ---
            result = await ai_writer.generate_and_optionally_test(
                requirement=requirement_text,
                human_tests_input=human_tests_data # Pass the raw list of dicts
            )
            # --- Format response based on result ---
            # (Similar formatting as before, but adapt based on new result structure)
            if result["success"]:
                logger.info("AI writer succeeded.")
                response_text = f"Successfully generated function based on requirement.\n\n"
                response_text += f"Specification:\n{json.dumps(result.get('spec'), indent=2)}\n\n"
                if result.get("test_run_results"):
                    response_text += f"Test Results Summary: Passed={result['test_run_results'].get('passed',0)}, Failed={result['test_run_results'].get('failed',0)}, Errors={result['test_run_results'].get('errors',0)}\n\n"
                else:
                    response_text += "Testing was not requested or performed.\n\n"
                response_text += f"Generated Code:\n```python\n{result['code']}\n```"
                return [types.TextContent(type="text", text=response_text)]
            else:
                # ... (Error formatting as before) ...
                logger.error(f"AI writer failed. Errors: {result['errors']}")
                error_details = "\n".join(f"- {e}" for e in result["errors"])
                response_text = f"Failed to generate function.\nErrors:\n{error_details}"
                # ... (include partial results if available) ...
                return [types.TextContent(type="text", text=response_text)]

        except Exception as e:
            logger.exception(f"Unexpected error calling AI writer within tool: {e}")
            return [types.TextContent(type="text", text=f"An internal server error occurred: {e}")]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """Lists the available tools."""
        return [
            types.Tool(
                name="create_python_function",
                description=(
                    "Generates a Python function specification and executable code based on a natural language 'requirement'. "
                    "Optionally, if a 'test_cases' list (containing human-written `{'inputs': {...}, 'expected': ...}` objects) "
                    "is provided, the generated code will be validated against these tests. "
                    "IMPORTANT: The 'test_cases' argument should ONLY be used with human-validated tests; the calling AI should NOT generate them. "
                    "If tests are provided, the keys in the 'inputs' dictionary will determine the exact parameter names of the generated function."
                ),
                inputSchema=CREATE_FUNCTION_INPUT_SCHEMA,
            )
        ]

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send # Use request._send for Starlette >= 0.35.0? Check library docs if needed.
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        starlette_app = Starlette(
            debug=True, # Set to False in production
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        try:
            import uvicorn
            uvicorn.run(starlette_app, host="0.0.0.0", port=port)
        except ImportError:
            print("Error: 'uvicorn' is not installed. Please install it to use SSE transport.")
            print("Install with: pip install uvicorn")
            return 1
        except Exception as e:
             print(f"Error running SSE server: {e}")
             return 1

    else: # stdio transport
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        try:
            anyio.run(arun)
        except Exception as e:
            print(f"Error running stdio server: {e}")
            return 1

    return 0