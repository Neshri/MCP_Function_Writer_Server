import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import json
# Import the specific types for better checking (optional but good practice)
# from mcp import types as mcp_types

# Make sure this client script is saved as MCP_client.py or similar
# in the root of your project directory (where pyproject.toml is)

async def main():
    print("Attempting to connect to MCP server via stdio...")
    try:
        async with stdio_client(
            # Make sure the command matches how you installed/run the server
            StdioServerParameters(command="uv", args=["run", "mcp-function-generator"])
        ) as (read, write):
            print("Connection established. Initializing session...")
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("Session initialized.")

                # List available tools
                print("Listing available tools...")
                list_tools_response = await session.list_tools()
                print("Available Tools Found:")
                # print(f"DEBUG: Raw list_tools response type: {type(list_tools_response)}")
                # print(f"DEBUG: Raw list_tools response data: {list_tools_response}")

                actual_tools_list = []
                # --- Access the .tools attribute directly ---
                if hasattr(list_tools_response, 'tools') and isinstance(list_tools_response.tools, list):
                    actual_tools_list = list_tools_response.tools
                else:
                    print(f"  Warning: Could not find '.tools' list attribute in response.")
                    print(f"  Response type: {type(list_tools_response)}")
                    print(f"  Response data: {list_tools_response}")


                if not actual_tools_list:
                    print("  (No tools found in the response's 'tools' attribute)")

                # --- Iterate through the actual list of tools ---
                for tool in actual_tools_list:
                    try:
                        # --- Use attribute access ---
                        print(f"- {tool.name}: {tool.description}")
                        # Pretty print the schema dictionary
                        print(f"  Input Schema: {json.dumps(tool.inputSchema, indent=4)}")
                    except AttributeError as e:
                         print(f"  Error accessing attributes on tool object: {tool}. Exception: {e}")
                    except Exception as e:
                         print(f"  Unexpected error processing tool: {tool}. Exception: {e}")


                # Define the function requirements
                function_details = {
  "requirement": "Problem: Check if a number is prime...",
  "test_cases": [
    {"inputs": {"number": 2}, "expected": True},
    {"inputs": {"number": 7}, "expected": True},
    {"inputs": {"number": 10}, "expected": False},
    {"inputs": {"number": 1}, "expected": False},
    {"inputs": {"number": 15}, "expected": False},
    {"inputs": {"number": 97}, "expected": True}
  ]
}

                print(f"\nCalling tool 'create_python_function' with args:\n{json.dumps(function_details, indent=2)}")

                # Call the tool
                call_tool_response = await session.call_tool("create_python_function", function_details)
                # print(f"DEBUG: Raw call_tool response type: {type(call_tool_response)}")
                # print(f"DEBUG: Raw call_tool response data: {call_tool_response}")

                print("\nResult:")

                actual_content_list = []
                # --- Access the .content attribute directly ---
                if hasattr(call_tool_response, 'content') and isinstance(call_tool_response.content, list):
                     actual_content_list = call_tool_response.content
                else:
                     print(f"  Warning: Could not find '.content' list attribute in response.")
                     print(f"  Response type: {type(call_tool_response)}")
                     print(f"  Response data: {call_tool_response}")


                if not actual_content_list:
                     print("  (No content found in the response's 'content' attribute)")

                # --- Iterate through the actual list of content blocks ---
                for content in actual_content_list:
                    try:
                        # --- Use attribute access for TextContent ---
                        if hasattr(content, 'type') and content.type == "text":
                             print("--- Generated Code Start ---")
                             print(content.text)
                             print("--- Generated Code End ---")
                        else:
                             print(f"Received non-Text content or unexpected structure: {content}")
                    except AttributeError as e:
                         print(f"  Error accessing attributes on content object: {content}. Exception: {e}")
                    except Exception as e:
                         print(f"  Unexpected error processing content: {content}. Exception: {e}")


            print("Session closed.")

    except Exception as e:
        import traceback
        print("\n--- An error occurred during client execution ---")
        traceback.print_exc()
        print("--------------------------------------------------")
    finally:
        # await asyncio.sleep(0.1) # Keep commented unless pipe errors are problematic
        print("Client script finished.")


if __name__ == "__main__":
    # Ensure the server is running in a separate terminal before executing this client
    # Command: uv run mcp-function-generator
    asyncio.run(main())