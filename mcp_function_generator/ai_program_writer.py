import json
import datetime
import re
import traceback
import time
import logging
import logging.handlers
import os
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party libraries
import ollama
import anyio # Required for async sleep and eventual MCP integration
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ValidationError

# --- Configuration ---

class AISettings(BaseSettings):
    """Loads settings from environment variables or .env file."""
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    debug_logging: bool = Field(True, alias='DEBUG_LOGGING')

    # Model configuration
    spec_model: str = Field("deepseek-coder:6.7b", alias='SPEC_MODEL')
    code_model: str = Field("deepseek-coder:6.7b", alias='CODE_MODEL')
    test_model: str = Field("deepseek-coder:6.7b", alias='TEST_MODEL')

    # Retry configuration
    spec_retries: int = Field(3, alias='SPEC_RETRIES')
    code_retries: int = Field(3, alias='CODE_RETRIES')
    test_retries: int = Field(3, alias='TEST_RETRIES')

    # Generation parameters (can be overridden via env)
    spec_temperature: float = Field(0.0, alias='SPEC_TEMPERATURE')
    code_temperature: float = Field(0.1, alias='CODE_TEMPERATURE')
    test_temperature: float = Field(0.1, alias='TEST_TEMPERATURE')
    default_seed: int = Field(42) # Seed might need variation per attempt sometimes


# --- Custom Exceptions ---

class AIWriterError(Exception):
    """Base exception for AI writer errors."""
    # You can add optional arguments to store more context if needed
    def __init__(self, message, *args, **kwargs):
        super().__init__(message, *args)
        # Store extra context if provided
        self.details = kwargs

class SpecGenerationError(AIWriterError):
    """Error during specification generation."""
    pass # Inherits __init__ from AIWriterError or Exception

class TestGenerationError(AIWriterError):
    """Error during test case generation."""
    pass # Inherits __init__ from AIWriterError or Exception

class CodeGenerationError(AIWriterError):
    """Error during code generation attempts."""
    # Example of adding specific attributes
    def __init__(self, message, *args, last_code=None, last_test_results=None, last_feedback=None, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.last_code = last_code
        self.last_test_results = last_test_results
        self.last_feedback = last_feedback

class CodeExecutionError(AIWriterError):
    """Error executing generated code (syntax, runtime)."""
    # Example of adding specific attributes
    def __init__(self, message, *args, code=None, traceback=None, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.code = code
        self.traceback = traceback

class TestRunError(AIWriterError):
    """Error running tests against generated code."""
     # Example of adding specific attributes
    def __init__(self, message, *args, results=None, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.results = results


# --- Data Structures ---

class TestCase:
    def __init__(self, function_name: str, inputs: Dict[str, Any], expected: Any) -> None:
        self.function_name = function_name
        self.inputs = inputs
        self.expected = expected

    def __str__(self) -> str:
        return f"TestCase(func='{self.function_name}', inputs={self.inputs}, expected={self.expected})"


class FunctionSpec:
    def __init__(self, name: str, parameters: Dict[str, str], docstring: str, returns: str) -> None:
        # Basic validation
        if not name or not isinstance(name, str):
            raise ValueError("Function name must be a non-empty string")
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary")
        # Add more validation if needed
        self.name = name
        self.parameters = parameters
        self.docstring = docstring
        self.returns = returns

    def __str__(self) -> str:
        return (f"FunctionSpec(name='{self.name}', parameters={self.parameters}, "
                f"docstring='{self.docstring}', returns='{self.returns}')")


# --- Logging Setup ---


logger = logging.getLogger("AIProgramWriter")


# --- Core Class ---

class AIProgramWriter:
    """
    Generates and validates Python code based on a natural language requirement
    using Ollama models.
    """

    # --- Prompt Templates ---
    SPEC_PROMPT_TEMPLATE = """SYSTEM: You are a precise code specification generator. Respond ONLY with valid JSON fitting the template. No explanations, apologies, or introductory text. Ensure types are valid Python type hints. The docstring should concisely describe the function's core purpose.
REQUIREMENT:
{requirement}

JSON TEMPLATE:
{{
    "name": "function_name",
    "parameters": {{ // Mapping of parameter name to Python type hint string (e.g., "s": "str", "nums": "list[int]")
        "param_name_1": "type_hint_1",
        "param_name_2": "type_hint_2"
    }},
    "docstring": "Concise description (under 20 words) of what the function does.",
    "returns": "Python type hint string for the return value (e.g., 'str', 'bool', 'Optional[int]')"
}}

RULES:
1. Output *only* the JSON object.
2. Parameter and return types must be valid Python type hints (e.g., `str`, `int`, `list[int]`, `dict[str, float]`, `Optional[bool]`).
3. The docstring must accurately and concisely reflect the requirement.
4. Do not use markdown fences like ```json.

VALID JSON RESPONSE:"""

    CODE_PROMPT_TEMPLATE = """SYSTEM: You are a Python code generation assistant. Create a Python function based *only* on the provided specification. Include necessary imports if standard libraries are needed (like `math`, `re`, `typing`). Handle potential edge cases and errors gracefully where appropriate (e.g., empty inputs, type mismatches if feasible within the function). Respond ONLY with the complete Python code for the function definition, including imports if needed. Do NOT include example usage, explanations outside comments, or markdown fences.

SPECIFICATION:
Function Name: {name}
Parameters: {parameters}
Return Type: {returns}
Description: {docstring}
{error_feedback_section}
PYTHON CODE ONLY:"""

    TEST_CASE_PROMPT_TEMPLATE = """SYSTEM: You are a test case generator. Create a diverse set of relevant test cases (including edge cases) for the given Python function specification. Respond ONLY with a valid JSON object containing a list under the "tests" key, following the exact format provided. Do not include explanations, apologies, or text outside the JSON structure.

FUNCTION SPECIFICATION:
Name: {name}
Parameters: {parameters}
Return Type: {returns}
Description: {docstring}
{error_feedback_section}
JSON TEMPLATE:
{{
    "tests": [
        {{ "inputs": {{ "param_name_1": <value_1>, "param_name_2": <value_2> }}, "expected": <expected_return_value> }},
        // ... more test cases ...
    ]
}}

RULES:
1. Provide at least 3 diverse test cases (normal, edge, potentially invalid if relevant to error handling).
2. Ensure input values match the expected types in the parameters (e.g., if "nums": "list[int]", provide `[1, 2, 3]`, not `"1, 2, 3"`).
3. Ensure the "expected" value matches the specified return type.
4. Output *only* the JSON object. Do not use markdown fences.

VALID JSON RESPONSE:"""

    def __init__(self, settings: AISettings) -> None:
        self.settings = settings
        self._setup_logging() # Call the setup method
        logger.info(f"AIProgramWriter initialized with settings: {settings.model_dump()}")

    def _setup_logging(self) -> None:
        """Configures logging based on settings, including file output."""
        log_level = logging.DEBUG if self.settings.debug_logging else logging.INFO
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        log_instance = logging.getLogger("AIProgramWriter") # Get logger instance again
        log_instance.setLevel(log_level) # Set minimum level for this logger instance

        # --- Crucial: Clear existing handlers to avoid duplication on re-init ---
        # This is important if the script/server might re-instantiate the class
        if log_instance.hasHandlers():
             log_instance.handlers.clear()

        # --- Console Handler (Optional - for direct runs) ---
        # Keep this if you want logs during direct `python ai_program_writer.py` runs
        # It might not show up when run via MCP client/uv run.
        console_handler = logging.StreamHandler() # Defaults to sys.stderr
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(log_level) # Set level for this handler
        log_instance.addHandler(console_handler)

        # --- File Handler (Persistent Log) ---
        log_filepath = None
        try:
            # Log to a file in the *project root* (where pyproject.toml is)
            # Assumes ai_program_writer.py is in a subdirectory like 'mcp_function_generator'
            log_filename = "ai_writer_interactions.log"
            # Construct path relative to the script's parent directory (project root)
            # This is slightly fragile; a better way might pass project_root in
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir) # Assumes script is one level down
            log_filepath = os.path.join(project_root, log_filename)

            file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8') # Append mode
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(logging.DEBUG) # Log EVERYTHING to the file if debug is enabled overall
            log_instance.addHandler(file_handler)
            # Use print here initially as logging might be bootstrapping
            print(f"[AIWriter Setup] Logging configured. Level={logging.getLevelName(log_level)}. Log file='{log_filepath}'")
        except Exception as e:
             print(f"[AIWriter Setup] Error setting up file logger for '{log_filepath}': {e}")

        # Prevent logs from propagating to the root logger if desired
        # log_instance.propagate = False
    async def generate_and_validate(self, requirement: str) -> Dict[str, Any]:
        """
        Main orchestration method to generate spec, tests, and validated code.
        """
        result: Dict[str, Any] = {"success": False, "code": None, "spec":None, "test_cases": None, "test_run_results": None, "errors": []}
        start_time = time.perf_counter()
        logger.info(f"Starting generation for requirement: '{requirement[:100]}...'")

        try:
            # 1. Generate Specification
            spec = await self._create_spec_with_retries(requirement)
            result["spec"] = spec.__dict__ if spec else None # Store spec details
            logger.info(f"Specification generated: {spec}")

            # 2. Generate Test Cases
            test_cases = await self._generate_test_cases_with_retries(spec)
            result["test_cases"] = [tc.__dict__ for tc in test_cases] # Store test case details
            logger.info(f"Generated {len(test_cases)} test cases.")

            # 3. Generate and Validate Code
            code, test_run_results = await self._generate_and_test_code_with_retries(spec, test_cases)
            result["code"] = code
            result["test_run_results"] = test_run_results # Store final test results
            result["success"] = True
            logger.info("Code generation and validation successful.")

        except AIWriterError as e:
            logger.error(f"Generation failed: {e}", exc_info=self.settings.debug_logging)
            result["errors"].append(str(e))
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}") # Logs full traceback
            result["errors"].append(f"Unexpected error: {e}")
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"Generation process finished in {elapsed:.2f} ms. Success: {result['success']}")
            return result


    async def _ollama_generate_stream(self, model: str, prompt: str, options: Dict[str, Any]) -> str:
        """Helper to call Ollama and stream response, handling potential errors."""
        full_response = ""
        logger.debug(f"Ollama Request: Model={model}, Options={options}\n--- Prompt ---\n{prompt}\n-------------")
        try:
            # --- ASYNC NOTE ---
            # If ollama.generate is synchronous (check its docs/source), wrap it for async:
            # stream = await anyio.to_thread.run_sync(
            #     ollama.generate,
            #     model=model,
            #     prompt=prompt,
            #     stream=True,
            #     options=options,
            #     cancellable=True # Good practice with run_sync
            # )
            # Assuming ollama.generate *might* be async compatible or for PoC:
            stream = ollama.generate(
                model=model,
                prompt=prompt,
                stream=True,
                options=options,
            )
            # --- End Async Note ---

            print(f"\n[{model} Stream] ", end="", flush=True) # User feedback
            for chunk in stream:
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full_response += token
            print() # Newline after stream
        except Exception as e:
            logger.error(f"Ollama API call failed for model {model}: {e}", exc_info=self.settings.debug_logging)
            raise AIWriterError(f"Ollama API error for {model}: {e}") from e

        logger.debug(f"Ollama Raw Response:\n{full_response}")
        return full_response.strip()

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempts to extract a JSON object from LLM response text."""
        logger.debug("Attempting JSON extraction...")
        try:
            # Prioritize fenced code blocks
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
            if match:
                logger.debug("Found JSON in fenced code block.")
                return json.loads(match.group(1))

            # Try direct parsing (if model behaves perfectly)
            try:
                logger.debug("Attempting direct JSON parsing.")
                return json.loads(text)
            except json.JSONDecodeError:
                logger.debug("Direct JSON parsing failed.")
                pass # Continue to regex fallback

            # Fallback: find first '{' and last '}' - risky!
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                logger.warning("Using potentially unreliable regex fallback for JSON extraction.")
                return json.loads(match.group(1))

            logger.error("No valid JSON object found in the response.")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}. Response segment: '{text[:200]}...'")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during JSON extraction: {e}", exc_info=True)
            return None

    def _extract_code(self, text: str) -> str:
        """Attempts to extract Python code, prioritizing fenced blocks."""
        logger.debug("Attempting Python code extraction...")
        # Prioritize ```python ... ```
        match_python = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match_python:
            logger.debug("Extracted code from ```python block.")
            return match_python.group(1).strip()

        # Fallback to generic ``` ... ```
        match_generic = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match_generic:
            logger.warning("Extracted code from generic ``` block.")
            return match_generic.group(1).strip()

        # Final fallback: assume the whole text might be code (if prompt requests ONLY code)
        logger.warning("No fenced code block found, returning entire response as potential code.")
        return text.strip() # Return the whole cleaned text

    async def _create_spec_with_retries(self, requirement: str) -> FunctionSpec:
        """Generates FunctionSpec with retries and exponential backoff."""
        logger.info("Attempting to generate function specification...")
        last_error = None
        options = {
            "temperature": self.settings.spec_temperature,
            "seed": self.settings.default_seed, # Consider varying seed per retry?
            "top_k": 1 # More deterministic for spec
        }
        prompt = self.SPEC_PROMPT_TEMPLATE.format(requirement=requirement)

        for attempt in range(self.settings.spec_retries):
            if last_error:
                logger.warning(f"Retrying spec generation (Attempt {attempt + 1}/{self.settings.spec_retries}) due to: {last_error}")
                await anyio.sleep(1 * (2**attempt)) # Exponential backoff

            try:
                full_response = await self._ollama_generate_stream(self.settings.spec_model, prompt, options)
                if not full_response:
                    raise AIWriterError("Received empty response from spec model.")

                data = self._extract_json(full_response)
                if not data:
                    raise SpecGenerationError("Failed to extract valid JSON specification.")

                spec = FunctionSpec(**data) # Pydantic performs basic validation here
                return spec # Success

            except (AIWriterError, ValidationError, ValueError, KeyError) as e:
                last_error = f"Attempt {attempt + 1}: {e}"
                logger.debug(f"Spec generation attempt {attempt + 1} failed: {e}")

        logger.error(f"Specification generation failed after {self.settings.spec_retries} attempts.")
        raise SpecGenerationError(f"Failed to generate specification after multiple retries. Last error: {last_error}")


    async def _generate_test_cases_with_retries(self, spec: FunctionSpec) -> List[TestCase]:
        """Generates test cases with retries and exponential backoff."""
        logger.info("Attempting to generate test cases...")
        last_error = None
        options = {
            "temperature": self.settings.test_temperature,
            "seed": self.settings.default_seed, # Consider varying seed
        }

        for attempt in range(self.settings.test_retries):
            error_feedback_section = f"\nPREVIOUS ERROR (Attempt {attempt+1}):\n{last_error}" if last_error else ""
            prompt = self.TEST_CASE_PROMPT_TEMPLATE.format(
                name=spec.name,
                parameters=spec.parameters,
                returns=spec.returns,
                docstring=spec.docstring,
                error_feedback_section=error_feedback_section
            )

            if last_error:
                logger.warning(f"Retrying test case generation (Attempt {attempt + 1}/{self.settings.test_retries}) due to: {last_error}")
                await anyio.sleep(1 * (2**attempt))

            try:
                full_response = await self._ollama_generate_stream(self.settings.test_model, prompt, options)
                if not full_response:
                    raise AIWriterError("Received empty response from test model.")

                data = self._extract_json(full_response)
                if not data:
                    raise TestGenerationError("Failed to extract valid JSON for test cases.")
                if "tests" not in data or not isinstance(data["tests"], list):
                    raise TestGenerationError("Missing or invalid 'tests' list in JSON.")

                test_cases = []
                for i, test_data in enumerate(data["tests"]):
                     if not isinstance(test_data, dict) or "inputs" not in test_data or "expected" not in test_data:
                         logger.warning(f"Skipping invalid test case data at index {i}: {test_data}")
                         continue
                     test_cases.append(TestCase(spec.name, test_data["inputs"], test_data["expected"]))

                if not test_cases:
                    raise TestGenerationError("No valid test cases could be parsed from the response.")

                logger.info(f"Successfully parsed {len(test_cases)} test cases.")
                return test_cases # Success

            except (AIWriterError, ValidationError, ValueError, KeyError, TypeError) as e:
                last_error = f"Attempt {attempt + 1}: {e}"
                logger.debug(f"Test case generation attempt {attempt + 1} failed: {e}")

        logger.error(f"Test case generation failed after {self.settings.test_retries} attempts.")
        raise TestGenerationError(f"Failed to generate test cases after multiple retries. Last error: {last_error}")


    async def _attempt_code_generation(self, spec: FunctionSpec, attempt: int, feedback: Optional[str]) -> str:
        """Single attempt to generate code, returning raw code string."""
        logger.info(f"Attempting code generation (Attempt {attempt + 1}/{self.settings.code_retries})...")
        error_feedback_section = f"\nPREVIOUS ERROR (Attempt {attempt+1}):\n{feedback}" if feedback else ""
        prompt = self.CODE_PROMPT_TEMPLATE.format(
            name=spec.name,
            parameters=spec.parameters,
            returns=spec.returns,
            docstring=spec.docstring,
            error_feedback_section=error_feedback_section
        )
        options = {
            "temperature": self.settings.code_temperature,
            "seed": self.settings.default_seed + attempt, # Vary seed per attempt
        }

        full_response = await self._ollama_generate_stream(self.settings.code_model, prompt, options)
        if not full_response:
            raise AIWriterError("Received empty response from code model.")

        code = self._extract_code(full_response)
        if not code:
             raise AIWriterError("Failed to extract any code from the response.")
        logger.debug(f"Code generation attempt {attempt+1} result:\n{code}")
        return code


    def _validate_code_syntax_and_signature(self, code: str, spec: FunctionSpec) -> None:
        """
        Checks if code is syntactically valid Python and if the function exists.
        Raises CodeExecutionError on failure.

        !!! SECURITY WARNING !!!
        This function uses exec(), which is inherently insecure if the code
        can come from untrusted sources (like an LLM). Execute this only in
        sandboxed environments or if you fully trust the source and understand
        the risks. It can execute arbitrary code with the permissions of the
        Python process.
        """
        logger.warning("Executing _validate_code_syntax_and_signature which uses potentially unsafe exec().")
        logger.debug(f"Validating syntax for code:\n---\n{code}\n---") # Log the code being exec'd
        namespace = {}
        try:
            # --- ASYNC NOTE ---
            # exec() is synchronous and CPU-bound. In an async context, run it in a thread:
            # await anyio.to_thread.run_sync(exec, code, namespace)
            exec(code, namespace)
            # --- End Async Note ---
            logger.debug("Code syntax validation passed (exec completed without error).")
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            # --- CORRECTED LINE ---
            raise CodeExecutionError(f"Syntax Error: {e.msg} (Line: {e.lineno}, Offset: {e.offset})", code=code, traceback=traceback.format_exc()) from e
            # --- END CORRECTION ---
        except Exception as e:
            logger.error(f"Unexpected error during code execution (exec): {e}", exc_info=self.settings.debug_logging)
            raise CodeExecutionError(f"Execution Error during validation: {e}", code=code, traceback=traceback.format_exc()) from e

        # Check if the function was defined
        if spec.name not in namespace:
            logger.error(f"Function '{spec.name}' not found in executed code namespace.")
            raise CodeExecutionError(f"Function '{spec.name}' was not defined in the generated code.", code=code)

        # Optional: Basic signature check (can be complex)
        try:
            generated_func = namespace[spec.name]
            signature = inspect.signature(generated_func)
            param_names = list(signature.parameters.keys())
            expected_params = list(spec.parameters.keys())
            # This is a basic check; doesn't validate types or defaults deeply
            if param_names != expected_params:
                 logger.warning(f"Parameter mismatch: Expected {expected_params}, Got {param_names}")
                 # Decide if this should be an error or just a warning
                 # raise CodeExecutionError(f"Parameter mismatch: Expected {expected_params}, Got {param_names}", code=code)
        except Exception as e:
            logger.warning(f"Could not perform signature inspection: {e}")

        logger.info(f"Code syntax and function '{spec.name}' presence validated.")


    def _run_tests(self, code: str, spec: FunctionSpec, test_cases: List[TestCase]) -> Dict[str, Any]:
        """
        Executes the generated code and runs test cases against it.
        Returns detailed test results. Raises TestRunError on exec failure.

        !!! SECURITY WARNING !!!
        This function uses exec(), which is inherently insecure. See warning
        in _validate_code_syntax_and_signature.
        """
        logger.warning("Executing _run_tests which uses potentially unsafe exec().")
        logger.info(f"Running {len(test_cases)} tests against generated code...")
        results: Dict[str, Any] = {"passed": 0, "failed": 0, "errors": 0, "details": []}
        namespace = {}

        try:
            # --- ASYNC NOTE ---
            # exec() is synchronous and CPU-bound. In an async context, run it in a thread:
            # await anyio.to_thread.run_sync(exec, code, namespace)
            exec(code, namespace)
            # --- End Async Note ---
        except Exception as e:
            logger.error(f"Fatal error executing code before running tests: {e}", exc_info=self.settings.debug_logging)
            # If exec fails, all tests error out
            results["errors"] = len(test_cases)
            error_detail = {"status": "error", "error": f"Code execution failed: {e}"}
            results["details"] = [error_detail] * len(test_cases)
            # Raise or return? Raising gives clearer signal up the chain.
            raise TestRunError(f"Code execution failed before tests could run: {e}", results=results) from e

        if spec.name not in namespace:
             # This should have been caught by validation, but double-check
             logger.error(f"Function '{spec.name}' not found in namespace during test run.")
             raise TestRunError(f"Function '{spec.name}' not in namespace.", results=results)

        func_to_test = namespace[spec.name]

        for i, test in enumerate(test_cases):
            test_detail = {"test_number": i + 1, "inputs": test.inputs, "expected": test.expected}
            logger.debug(f"Running test {i+1}: Input={test.inputs}, Expected={test.expected}")
            try:
                # --- ASYNC NOTE ---
                # If the generated function itself is async, you'd need to await it here.
                # If it's synchronous (likely), run it in a thread if it might block.
                # actual = await anyio.to_thread.run_sync(func_to_test, **test.inputs)
                actual = func_to_test(**test.inputs)
                # --- End Async Note ---

                # Basic comparison. Might need adjustment for floats, complex objects.
                if actual == test.expected:
                    results["passed"] += 1
                    test_detail["status"] = "passed"
                    test_detail["actual"] = actual # Include actual even on pass
                    logger.debug(f"Test {i+1} passed.")
                else:
                    results["failed"] += 1
                    test_detail["status"] = "failed"
                    test_detail["actual"] = actual
                    logger.warning(f"Test {i+1} failed: Expected={test.expected}, Actual={actual}")
            except Exception as e:
                results["errors"] += 1
                test_detail["status"] = "error"
                test_detail["error"] = str(e)
                logger.error(f"Test {i+1} raised an error: {e}", exc_info=self.settings.debug_logging)

            results["details"].append(test_detail)

        logger.info(f"Tests completed: {results['passed']} passed, {results['failed']} failed, {results['errors']} errors.")
        return results


    async def _generate_and_test_code_with_retries(self, spec: FunctionSpec, test_cases: List[TestCase]) -> Tuple[str, Dict[str, Any]]:
        """Orchestrates the code generation, validation, and testing loop."""
        logger.info("Starting code generation and testing loop...")
        feedback: Optional[str] = None
        last_successful_code: Optional[str] = None
        final_test_results: Optional[Dict[str, Any]] = None

        for attempt in range(self.settings.code_retries):
            if feedback:
                 logger.warning(f"Retrying code generation (Attempt {attempt + 1}/{self.settings.code_retries}) with feedback.")
                 await anyio.sleep(1 * (2**attempt)) # Exponential backoff only needed on retry

            try:
                # 1. Attempt to generate code
                code = await self._attempt_code_generation(spec, attempt, feedback)
                last_successful_code = code # Store latest generated code

                # 2. Validate Syntax & Function Existence (Raises CodeExecutionError on failure)
                # Note: Contains exec() - see safety warnings
                self._validate_code_syntax_and_signature(code, spec)

                # 3. Run Tests (Raises TestRunError on exec failure, returns results otherwise)
                 # Note: Contains exec() - see safety warnings
                test_run_results = self._run_tests(code, spec, test_cases)
                final_test_results = test_run_results # Store latest test results

                if test_run_results["failed"] == 0 and test_run_results["errors"] == 0:
                    logger.info(f"Code passed all {len(test_cases)} tests on attempt {attempt + 1}.")
                    return code, test_run_results # SUCCESS!

                # 4. Prepare feedback for next attempt based on test results
                failed_details = [d for d in test_run_results["details"] if d["status"] != "passed"]
                feedback = f"Code executed but failed {len(failed_details)} test(s):\n"
                for detail in failed_details[:3]: # Limit feedback length
                    if detail['status'] == 'failed':
                        feedback += f"- Input: {detail['inputs']}, Expected: {detail['expected']}, Got: {detail['actual']}\n"
                    elif detail['status'] == 'error':
                         feedback += f"- Input: {detail['inputs']}, Expected: {detail['expected']}, Error: {detail['error']}\n"
                logger.warning(f"Attempt {attempt+1} failed tests. Feedback for next attempt: {feedback.strip()}")


            except CodeExecutionError as e:
                logger.error(f"Code validation/execution failed on attempt {attempt + 1}: {e}")
                feedback = f"Code execution failed: {e}" # Feedback is the execution error
                final_test_results = None # Reset test results as code didn't run properly

            except TestRunError as e:
                 logger.error(f"Test run environment failed on attempt {attempt + 1}: {e}")
                 feedback = f"Code execution failed during test setup: {e}" # Feedback is the setup error
                 final_test_results = e.results if hasattr(e, 'results') else None

            except AIWriterError as e:
                logger.error(f"Code generation attempt {attempt + 1} failed: {e}")
                feedback = f"Code generation failed: {e}" # Feedback is the generation error itself
                final_test_results = None


        logger.error(f"Code generation failed to produce passing code after {self.settings.code_retries} attempts.")
        # Return the last generated code and its test results, even if failing
        raise CodeGenerationError(
            "Failed to generate code that passes all tests after multiple retries.",
            last_code=last_successful_code,
            last_test_results=final_test_results,
            last_feedback=feedback
        )


# --- Main Execution Example ---

async def run_main():
    """Example of how to use the AIProgramWriter."""
    try:
        settings = AISettings() # type: ignore # Load from .env or defaults
    except ValidationError as e:
        print(f"Error loading settings: {e}")
        return

    writer = AIProgramWriter(settings)
    requirement = """Problem: Find the longest palindromic substring in a given string.
Example: Input: "babad" Output: "bab" or "aba"
Constraints: Input string contains only lowercase letters (a-z). String length is between 1 and 1000."""

    result = await writer.generate_and_validate(requirement)

    print("\n--- FINAL RESULT ---")
    if result["success"]:
        print("Generation Successful!")
        print("\nGenerated Specification:")
        print(json.dumps(result.get("spec"), indent=2))
        # print("\nGenerated Test Cases:")
        # print(json.dumps(result.get("test_cases"), indent=2)) # Can be verbose
        print("\nFinal Test Run Results:")
        print(json.dumps(result.get("test_run_results"), indent=2))
        print("\nGenerated Code:")
        print("```python")
        print(result["code"])
        print("```")
    else:
        print("Generation Failed.")
        print("\nErrors:")
        for error in result["errors"]:
            print(f"- {error}")
        if result.get("spec"):
             print("\nLast Generated Specification:")
             print(json.dumps(result.get("spec"), indent=2))
        if result.get("code"):
            print("\nLast Generated Code (May be faulty):")
            print("```python")
            print(result["code"])
            print("```")
        if result.get("test_run_results"):
            print("\nLast Test Run Results:")
            print(json.dumps(result.get("test_run_results"), indent=2))

if __name__ == "__main__":
    try:
        # Use anyio.run for the async main function
        anyio.run(run_main)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")
        traceback.print_exc()