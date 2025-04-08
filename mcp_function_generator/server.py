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
    ) -> list[types.TextContent]: # Or potentially return richer content
        tool_name = "create_python_function" # Or rename tool if you prefer
        if name != tool_name:
            raise ValueError(f"Unknown tool: {name}. Expected: '{tool_name}'")

        if "requirement" not in arguments:
            raise ValueError("Missing required argument 'requirement'")

        requirement_text = arguments["requirement"]

        # Instantiate writer here if not global
        # settings = AISettings()
        # ai_writer = AIProgramWriter(settings)

        logger.info(f"Received request for tool '{tool_name}' with requirement: {requirement_text[:100]}...")

        try:
            # --- ASYNC CALL ---
            # This is where the main generation happens
            result = await ai_writer.generate_and_validate(requirement_text)
            # --- END ASYNC CALL ---

            if result["success"]:
                logger.info("AI writer succeeded. Formatting positive response.")
                # Maybe format output better? Include summary?
                response_text = f"Successfully generated function based on requirement.\n\n"
                response_text += f"Specification:\n{json.dumps(result.get('spec'), indent=2)}\n\n"
                response_text += f"Test Results Summary: Passed={result.get('test_run_results',{}).get('passed',0)}, Failed={result.get('test_run_results',{}).get('failed',0)}, Errors={result.get('test_run_results',{}).get('errors',0)}\n\n"
                response_text += f"Generated Code:\n```python\n{result['code']}\n```"
                return [types.TextContent(type="text", text=response_text)]
            else:
                logger.error(f"AI writer failed. Formatting error response. Errors: {result['errors']}")
                error_details = "\n".join(f"- {e}" for e in result["errors"])
                # Include partial results if available?
                response_text = f"Failed to generate function.\nErrors:\n{error_details}"
                if result.get('code'):
                    response_text += f"\n\nLast attempted code (may be faulty):\n```python\n{result['code']}\n```"
                return [types.TextContent(type="text", text=response_text)]

        except Exception as e:
            # Catch unexpected errors during the call within the tool handler
            logger.exception(f"Unexpected error calling AI writer within tool: {e}")
            return [types.TextContent(type="text", text=f"An internal server error occurred: {e}")]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """Lists the available tools."""
        return [
            types.Tool(
                name="create_python_function",
                description="Generates Python code for a function based on descriptions of its name, arguments, return value, and logic.",
                inputSchema=CREATE_FUNCTION_INPUT_SCHEMA,
            )
        ]

    # --- Transport Handling (same as example, just ensure app is passed correctly) ---
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