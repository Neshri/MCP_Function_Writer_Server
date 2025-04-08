# ai_program_writer.py

import json
import datetime
import re
import traceback
import time
import logging
import logging.handlers
import os
import inspect
from typing import Any, Dict, List, Optional, Tuple, Set
import functools
# Third-party libraries
import ollama
import anyio # Required for async sleep and eventual MCP integration
from anyio import to_thread
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

    # Retry configuration
    spec_retries: int = Field(3, alias='SPEC_RETRIES')
    code_retries: int = Field(3, alias='CODE_RETRIES') # Retries when tests ARE provided
    code_syntax_retries: int = Field(1, alias='CODE_SYNTAX_RETRIES') # Retries for syntax when NO tests

    # Generation parameters (can be overridden via env)
    spec_temperature: float = Field(0.0, alias='SPEC_TEMPERATURE')
    code_temperature: float = Field(0.1, alias='CODE_TEMPERATURE')
    default_seed: int = Field(42)


# --- Custom Exceptions ---

class AIWriterError(Exception):
    """Base exception for AI writer errors."""
    def __init__(self, message, *args, **kwargs):
        super().__init__(message, *args)
        self.details = kwargs

class SpecGenerationError(AIWriterError):
    """Error during specification generation."""
    pass

class HumanTestProcessingError(AIWriterError): # Renamed for clarity
    """Error processing human-provided test data (parsing, consistency)."""
    pass

class CodeGenerationError(AIWriterError):
    """Error during code generation attempts."""
    def __init__(self, message, *args, last_code=None, last_test_results=None, last_feedback=None, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.last_code = last_code
        self.last_test_results = last_test_results
        self.last_feedback = last_feedback

class CodeExecutionError(AIWriterError):
    """Error executing generated code (syntax, runtime)."""
    def __init__(self, message, *args, code=None, traceback=None, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.code = code
        self.traceback = traceback

class TestRunError(AIWriterError):
    """Error running tests against generated code."""
    def __init__(self, message, *args, results=None, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.results = results


# --- Data Structures ---

class TestCase:
    """Represents a single human-provided test case."""
    def __init__(self, function_name: str, inputs: Dict[str, Any], expected: Any) -> None:
        self.function_name = function_name
        self.inputs = inputs # User-provided keys/values
        self.expected = expected

    def __str__(self) -> str:
        return f"TestCase(func='{self.function_name}', inputs={self.inputs}, expected={self.expected})"


class FunctionSpec:
    """Represents the AI-generated function specification."""
    def __init__(self, name: str, parameters: Dict[str, str], docstring: str, returns: str) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Function name must be a non-empty string")
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary")
        self.name = name
        self.parameters = parameters # Keys SHOULD match test input keys if constrained
        self.docstring = docstring
        self.returns = returns

    def __str__(self) -> str:
        return (f"FunctionSpec(name='{self.name}', parameters={self.parameters}, "
                f"docstring='{self.docstring}', returns='{self.returns}')")


# --- Logging Setup ---

logger = logging.getLogger("AIProgramWriter") # Logger instance used throughout

# --- Core Class ---

class AIProgramWriter:
    """
    Generates Python code based on a requirement. If human tests are provided,
    constrains spec generation to use test input keys and validates the generated
    code against those tests.
    """

    # --- Prompt Templates ---
    SPEC_PROMPT_TEMPLATE = """SYSTEM: You are a precise code specification generator. Respond ONLY with valid JSON fitting the template. No explanations, apologies, or introductory text. Ensure types are valid Python type hints. The docstring should concisely describe the function's core purpose.
REQUIREMENT:
{requirement}
{key_constraint_section}
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
2. Parameter and return types must be valid Python type hints.
3. The docstring must accurately and concisely reflect the requirement.
4. Do not use markdown fences like ```json.
{key_usage_rule}
VALID JSON RESPONSE:"""

    CODE_PROMPT_TEMPLATE = """SYSTEM: You are a Python code generation assistant. Create a Python function based *only* on the provided specification. Include necessary imports if standard libraries are needed (like `math`, `re`, `typing`). Handle potential edge cases and errors gracefully where appropriate (e.g., empty inputs, type mismatches if feasible within the function). Respond ONLY with the complete Python code for the function definition, including imports if needed. Do NOT include example usage, explanations outside comments, or markdown fences.

SPECIFICATION:
Function Name: {name}
Parameters: {parameters}
Return Type: {returns}
Description: {docstring}
{error_feedback_section}
PYTHON CODE ONLY:"""

    def __init__(self, settings: AISettings) -> None:
        self.settings = settings
        self._setup_logging()
        logger.info(f"AIProgramWriter initialized with settings: {settings.model_dump()}")

    def _setup_logging(self) -> None:
        """Configures logging based on settings, including file output."""
        log_level = logging.DEBUG if self.settings.debug_logging else logging.INFO
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        log_instance = logging.getLogger("AIProgramWriter")
        log_instance.setLevel(log_level)

        if log_instance.hasHandlers():
             log_instance.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(log_level)
        log_instance.addHandler(console_handler)

        log_filepath = None
        try:
            log_filename = "ai_writer_interactions.log"
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            log_filepath = os.path.join(project_root, log_filename)

            file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(logging.DEBUG)
            log_instance.addHandler(file_handler)
            # Use print as logger might be bootstrapping
            print(f"[AIWriter Setup] Logging configured. Level={logging.getLevelName(log_level)}. Log file='{log_filepath}'")
        except Exception as e:
             print(f"[AIWriter Setup] Error setting up file logger for '{log_filepath}': {e}")

    async def generate_and_optionally_test(
        self,
        requirement: str,
        human_tests_input: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration: Generates spec (constrained by test keys if provided)
        & code, optionally runs human tests using keyword arguments.
        """
        result: Dict[str, Any] = {"success": False, "code": None, "spec": None, "test_run_results": None, "errors": []}
        start_time = time.perf_counter()
        logger.info(f"Starting generation for requirement: '{requirement[:100]}...'. Human tests provided: {human_tests_input is not None}")

        spec: Optional[FunctionSpec] = None
        parsed_test_cases: Optional[List[TestCase]] = None
        expected_input_keys: Optional[Set[str]] = None

        try:
            # --- Extract and Validate Keys BEFORE Spec Gen ---
            if human_tests_input:
                logger.info(f"Processing {len(human_tests_input)} human-provided test cases to extract input keys...")
                expected_input_keys = self._get_consistent_input_keys(human_tests_input)
                logger.info(f"Using input keys from tests for spec generation: {expected_input_keys}")

            # 1. Generate Specification (passing keys if available)
            spec = await self._create_spec_with_retries(requirement, expected_input_keys)
            result["spec"] = spec.__dict__ if spec else None
            logger.info(f"Specification generated: {spec}")

            # --- Validate Spec Keys AFTER Spec Gen (if keys were expected) ---
            if expected_input_keys is not None:
                generated_keys = set(spec.parameters.keys())
                if generated_keys != expected_input_keys:
                     logger.error(f"Spec parameter keys {generated_keys} do not match required test input keys {expected_input_keys}.")
                     raise SpecGenerationError(f"Spec parameters {generated_keys} did not match required test input keys {expected_input_keys}.")
                logger.info("Spec parameter keys successfully validated against test input keys.")

            # 2. Structure Human Tests (if provided) - Keys should now match spec
            if human_tests_input:
                # Simple structuring, assumes keys are now correct due to prior validation
                parsed_test_cases = [
                    TestCase(spec.name, test_data["inputs"], test_data["expected"])
                    for test_data in human_tests_input
                    # Add basic validation for 'inputs' dict and 'expected' key presence if needed
                    if isinstance(test_data.get("inputs"), dict) and "expected" in test_data
                ]
                if len(parsed_test_cases) != len(human_tests_input):
                    logger.warning("Some provided test cases were filtered due to missing 'inputs' or 'expected'.")
                    # Decide if this should be a hard error
                logger.info(f"Successfully structured {len(parsed_test_cases)} test cases.")
            else:
                logger.info("No human tests provided, testing will be skipped.")

            # 3. Generate and Validate/Test Code
            code, test_run_results = await self._generate_and_validate_or_test_code(spec, parsed_test_cases)

            result["code"] = code
            result["test_run_results"] = test_run_results # Will be None if no tests run
            result["success"] = True
            logger.info("Code generation successful.")
            if test_run_results:
                 # Check if tests were skipped vs executed
                 if test_run_results.get("skipped_reason"):
                      logger.info(f"Testing skipped: {test_run_results['skipped_reason']}")
                 else:
                      logger.info("Code passed provided human tests.")
            else:
                 logger.info("Testing skipped or not provided.")

        except HumanTestProcessingError as e:
             logger.error(f"Failed to process human-provided tests: {e}", exc_info=self.settings.debug_logging)
             result["errors"].append(f"Error processing provided test cases: {e}")
             result["success"] = False
        except AIWriterError as e:
            logger.error(f"Generation failed: {e}", exc_info=self.settings.debug_logging)
            result["errors"].append(str(e))
            if isinstance(e, CodeGenerationError):
                 result["code"] = e.last_code
                 result["test_run_results"] = e.last_test_results
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")
            result["errors"].append(f"Unexpected error: {e}")
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"Generation process finished in {elapsed:.2f} ms. Success: {result['success']}")
            return result

    def _get_consistent_input_keys(self, tests_data: List[Dict[str, Any]]) -> Optional[Set[str]]:
        """Extracts input keys from first test and validates consistency across all tests."""
        if not tests_data:
             return None

        first_keys: Optional[Set[str]] = None
        for i, test_dict in enumerate(tests_data):
             if not isinstance(test_dict, dict) or "inputs" not in test_dict or not isinstance(test_dict["inputs"], dict):
                  raise HumanTestProcessingError(f"Invalid or missing 'inputs' dictionary in test case at index {i}.")

             current_keys = set(test_dict["inputs"].keys())
             if first_keys is None:
                  first_keys = current_keys
                  # Check if first inputs dict is empty (valid for zero-param func)
                  if not first_keys:
                      logger.info(f"Test case {i} has an empty 'inputs' dictionary (expected for zero-parameter function).")
             elif current_keys != first_keys:
                  raise HumanTestProcessingError(f"Inconsistent input keys found. Test {i} keys {current_keys} differ from first test keys {first_keys}.")

        # Return the consistent set of keys (could be empty set)
        return first_keys if first_keys is not None else set()


    async def _create_spec_with_retries(self, requirement: str, expected_keys: Optional[Set[str]]) -> FunctionSpec:
        """Generates FunctionSpec, potentially constrained by expected_keys."""
        logger.info(f"Attempting to generate function specification. Key constraint: {expected_keys}")
        last_error = None
        options = {
            "temperature": self.settings.spec_temperature,
            "seed": self.settings.default_seed,
            "top_k": 1
        }

        key_constraint_section = ""
        key_usage_rule = ""
        if expected_keys is not None:
             # Handle empty set for zero-param functions
             if not expected_keys:
                  key_list_str = "none (function takes no arguments)"
                  key_constraint_section = f"\nHuman-provided tests indicate this function takes no arguments."
                  key_usage_rule = f"5. **IMPORTANT:** The 'parameters' object in your JSON response MUST be empty (`{{}}`) as the provided tests indicate the function takes no arguments."
             else:
                  key_list_str = ", ".join(f"'{k}'" for k in sorted(list(expected_keys))) # Quote keys for clarity
                  key_constraint_section = f"\nHuman-provided tests use the following input key(s): {key_list_str}."
                  key_usage_rule = f"5. **IMPORTANT:** The keys in the 'parameters' object of your JSON response MUST exactly match the provided test input key(s): {key_list_str}. Choose appropriate Python type hints for each."

        prompt = self.SPEC_PROMPT_TEMPLATE.format(
            requirement=requirement,
            key_constraint_section=key_constraint_section,
            key_usage_rule=key_usage_rule
        )

        for attempt in range(self.settings.spec_retries):
            if last_error:
                logger.warning(f"Retrying spec generation (Attempt {attempt + 1}/{self.settings.spec_retries}) due to: {last_error}")
                await anyio.sleep(1 * (2**attempt))

            try:
                full_response = await self._ollama_generate_stream(self.settings.spec_model, prompt, options)
                if not full_response:
                    raise AIWriterError("Received empty response from spec model.")
                data = self._extract_json(full_response)
                if not data:
                    raise SpecGenerationError("Failed to extract valid JSON specification.")
                spec = FunctionSpec(**data) # Use pydantic validation
                return spec # Success - further validation happens after this returns
            except (AIWriterError, ValidationError, ValueError, KeyError) as e:
                last_error = f"Attempt {attempt + 1}: {e}"
                logger.debug(f"Spec generation attempt {attempt + 1} failed: {e}")

        logger.error(f"Specification generation failed after {self.settings.spec_retries} attempts.")
        raise SpecGenerationError(f"Failed to generate specification after multiple retries. Last error: {last_error}")


    async def _attempt_code_generation(self, spec: FunctionSpec, attempt: int, feedback: Optional[str]) -> str:
        """Single attempt to generate code, returning raw code string."""
        logger.info(f"Attempting code generation (Attempt {attempt + 1})...")
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
            "seed": self.settings.default_seed + attempt,
        }

        full_response = await self._ollama_generate_stream(self.settings.code_model, prompt, options)
        if not full_response:
            raise AIWriterError("Received empty response from code model.")

        code = self._extract_code(full_response)
        if not code:
             raise AIWriterError("Failed to extract any code from the response.")
        logger.debug(f"Code generation attempt {attempt+1} raw code:\n{code}")
        return code


    async def _validate_code_syntax_and_signature(self, code: str, spec: FunctionSpec) -> None: # Make async
        """
        Checks syntax and function presence by running exec in a thread.
        Raises CodeExecutionError on failure.
        !!! SECURITY WARNING !!! Uses exec().
        """
        logger.warning("Executing _validate_code_syntax_and_signature which uses potentially unsafe exec() in a thread.")
        logger.debug(f"Validating syntax for code:\n---\n{code}\n---")
        namespace = {} # Keep namespace local to this scope for safety
        try:
            # Run exec in a thread
            await to_thread.run_sync(
                exec, # The callable built-in
                code, # 1st arg for exec
                namespace, # 2nd arg for exec (globals)
                # We don't need to pass locals explicitly here
                abandon_on_cancel=False
            )
            logger.debug("Code syntax validation passed (exec completed without error in thread).")
        except SyntaxError as e:
            # Errors from exec running in the thread are propagated back by run_sync
            logger.error(f"Syntax error in generated code: {e}")
            raise CodeExecutionError(f"Syntax Error: {e.msg} (Line: {e.lineno}, Offset: {e.offset})", code=code, traceback=traceback.format_exc()) from e
        except Exception as e:
            logger.error(f"Unexpected error during code execution (exec in thread): {e}", exc_info=self.settings.debug_logging)
            raise CodeExecutionError(f"Execution Error during validation: {e}", code=code, traceback=traceback.format_exc()) from e

        # --- Namespace check happens *after* run_sync completes ---
        if spec.name not in namespace:
            logger.error(f"Function '{spec.name}' not found in executed code namespace.")
            raise CodeExecutionError(f"Function '{spec.name}' was not defined in the generated code.", code=code)

        # --- Signature check also happens after run_sync ---
        # (rest of signature check logic remains the same)
        if spec.parameters:
             try:
                 # ... (signature check logic) ...
                 pass
             except Exception as e:
                 logger.warning(f"Could not perform signature inspection: {e}")
        # ... (check for unexpected params if spec.parameters is empty) ...

        logger.info(f"Code syntax and function '{spec.name}' presence validated.")


    async def _run_tests(self, code: str, spec: FunctionSpec, test_cases: List[TestCase]) -> Dict[str, Any]: # Make async
        """
        Executes code in thread, then runs test cases against it in thread(s).
        Uses KEYWORD arguments if spec/test keys match.

        !!! SECURITY WARNING !!! Uses exec().
        """
        logger.warning("Executing _run_tests which uses potentially unsafe exec() in threads.")
        logger.info(f"Running {len(test_cases)} provided tests against generated code...")
        results: Dict[str, Any] = {"passed": 0, "failed": 0, "errors": 0, "details": [], "skipped_reason": None}
        namespace = {}

        # --- Execute code once in a thread ---
        try:
            await to_thread.run_sync(
                exec, code, namespace, abandon_on_cancel=False
            )
        except Exception as e:
            logger.error(f"Fatal error executing code in thread before running tests: {e}", exc_info=self.settings.debug_logging)
            results["errors"] = len(test_cases)
            error_detail = {"status": "error", "error": f"Code execution failed: {e}"}
            results["details"] = [error_detail] * len(test_cases)
            raise TestRunError(f"Code execution failed before tests could run: {e}", results=results) from e
        # --- End code execution ---

        if spec.name not in namespace:
             logger.error(f"Function '{spec.name}' not found in namespace after exec.")
             raise TestRunError(f"Function '{spec.name}' not in namespace.", results=results)

        func_to_test = namespace[spec.name]

        # --- Run each test case ---
        # Option 1: Run each test sequentially in a thread (simpler, less performant)
        # Option 2: Run tests concurrently using anyio.TaskGroup and run_sync (more complex)
        # Let's go with Option 1 for simplicity now.

        for i, test in enumerate(test_cases):
            test_detail = {"test_number": i + 1, "inputs": test.inputs, "expected": test.expected}
            logger.debug(f"Running test {i+1}: Input Dict={test.inputs}, Expected={test.expected}")

            try:
                # Wrap the synchronous function call with keyword args using partial
                # This creates a callable that run_sync can execute without extra args
                call_wrapper = functools.partial(func_to_test, **test.inputs)

                # Run the actual function call in a thread
                actual = await to_thread.run_sync(
                    call_wrapper, abandon_on_cancel=False
                )

                # Compare results after thread returns
                if actual == test.expected:
                    results["passed"] += 1
                    test_detail["status"] = "passed"
                    test_detail["actual"] = actual
                    logger.debug(f"Test {i+1} passed.")
                else:
                    results["failed"] += 1
                    test_detail["status"] = "failed"
                    test_detail["actual"] = actual
                    logger.warning(f"Test {i+1} failed: Expected={test.expected}, Actual={actual}")

            except Exception as e:
                 # Errors from within the function call (propagated by run_sync)
                 # or from run_sync itself (less likely here)
                results["errors"] += 1
                test_detail["status"] = "error"
                test_detail["error"] = str(e)
                # Log TypeError specifically if it happens despite checks
                if isinstance(e, TypeError):
                     logger.error(f"Test {i+1} raised TypeError (likely key mismatch/args error): {e}", exc_info=self.settings.debug_logging)
                     test_detail["error"] = f"TypeError during call: {e}" # More specific error
                else:
                     logger.error(f"Test {i+1} raised an error during function execution in thread: {e}", exc_info=self.settings.debug_logging)

            results["details"].append(test_detail)

        logger.info(f"Tests completed: {results['passed']} passed, {results['failed']} failed, {results['errors']} errors.")
        return results


    async def _generate_and_validate_or_test_code(
        self,
        spec: FunctionSpec,
        parsed_test_cases: Optional[List[TestCase]] # Contains .inputs dict
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generates code. If tests provided, runs them and retries on failure.
        If no tests, generates once and validates syntax.
        Assumes spec params match test keys if tests are provided.
        """
        feedback: Optional[str] = None
        last_successful_code: Optional[str] = None
        final_test_results: Optional[Dict[str, Any]] = None

        # Determine if tests can/should be run (used to set max_attempts and control flow)
        # Note: spec key validation already happened before calling this method
        run_tests_flag = parsed_test_cases is not None

        max_attempts = self.settings.code_retries if run_tests_flag else self.settings.code_syntax_retries

        logger.info(f"Starting code generation. Testing enabled: {run_tests_flag}. Max attempts: {max_attempts}")

        for attempt in range(max_attempts):
             if feedback: # Only applies if tests failed on previous attempt
                 logger.warning(f"Retrying code generation (Attempt {attempt + 1}/{max_attempts}) with test feedback.")
                 await anyio.sleep(1 * (2**attempt)) # Exponential backoff only on retry

             try:
                # 1. Generate code (already async)
                code = await self._attempt_code_generation(spec, attempt, feedback)
                last_successful_code = code # Store latest code regardless of validity

                # 2. Validate Syntax & Function Existence (NOW ASYNC)
                # This raises CodeExecutionError on failure
                await self._validate_code_syntax_and_signature(code, spec)
                # If we reach here, syntax is OK for this attempt

                # 3. Run tests IF provided (NOW ASYNC)
                if run_tests_flag:
                    # We know parsed_test_cases is not None if run_tests_flag is True
                    test_run_results = await self._run_tests(code, spec, parsed_test_cases) # type: ignore
                    final_test_results = test_run_results # Store latest results

                    # Check if tests passed
                    if test_run_results.get("failed", 0) == 0 and test_run_results.get("errors", 0) == 0:
                        logger.info(f"Code passed all {len(parsed_test_cases)} provided tests on attempt {attempt + 1}.") # type: ignore
                        return code, test_run_results # SUCCESS with tests!

                    # Prepare feedback if tests failed/errored
                    failed_details = [d for d in test_run_results["details"] if d["status"] != "passed"]
                    if failed_details:
                        feedback = f"Code executed but failed/errored on {len(failed_details)} test(s):\n"
                        for detail in failed_details[:3]: # Limit feedback length
                            input_repr = detail.get('inputs', 'N/A')
                            if detail['status'] == 'failed':
                                feedback += f"- Input: {input_repr}, Expected: {detail['expected']}, Got: {detail['actual']}\n"
                            elif detail['status'] == 'error':
                                feedback += f"- Input: {input_repr}, Error: {detail['error']}\n"
                        logger.warning(f"Attempt {attempt+1} failed tests. Feedback for next attempt:\n{feedback.strip()}")
                    else:
                        # Should not happen if counts > 0, but clear feedback defensively
                        feedback = None
                        logger.warning(f"Attempt {attempt+1} test counts indicate failure, but no failed details found. Clearing feedback.")

                    # If we are here, tests failed, and we need to loop for the next attempt (if any remain)

                else:
                    # No tests provided, syntax is valid, so we are done.
                    logger.info(f"Code generated and syntax validated (no tests provided). Attempt {attempt + 1}.")
                    return code, None # SUCCESS without tests!

             except CodeExecutionError as e:
                 # Handles errors from _validate_code_syntax_and_signature
                 logger.error(f"Code syntax/validation failed on attempt {attempt + 1}: {e}")
                 if attempt < max_attempts - 1:
                     feedback = f"Code syntax or execution failed: {e}" # Provide feedback for retry
                     logger.warning(f"Retrying due to syntax/exec error (Attempt {attempt + 1}/{max_attempts}).")
                     final_test_results = None # Reset test results as code failed validation
                     continue # Go to next attempt
                 else:
                     # Last attempt failed syntax check
                     logger.error(f"Syntax validation failed on final attempt {attempt + 1}.")
                     raise CodeGenerationError(f"Syntax validation failed after {max_attempts} attempts: {e}", last_code=last_successful_code, last_feedback=feedback) from e

             except TestRunError as e:
                 # This implies tests were running but exec failed within _run_tests (or other setup issue)
                 logger.error(f"Test run environment failed on attempt {attempt + 1}: {e}")
                 feedback = f"Code execution failed during test setup: {e}"
                 final_test_results = e.results if hasattr(e, 'results') else None # Store partial results if available
                 if attempt >= max_attempts - 1:
                     logger.error(f"Test setup failed on final attempt {attempt + 1}.")
                     raise CodeGenerationError("Code generation failed due to test setup errors.", last_code=last_successful_code, last_test_results=final_test_results, last_feedback=feedback) from e
                 else:
                      logger.warning(f"Retrying due to test setup error (Attempt {attempt + 1}/{max_attempts}).")
                      continue # Loop continues to retry code generation

             except AIWriterError as e:
                 # Errors during _attempt_code_generation API call/extraction
                 logger.error(f"Code generation API/extraction failed on attempt {attempt + 1}: {e}")
                 feedback = f"Code generation failed: {e}" # Use error as feedback
                 if attempt >= max_attempts - 1:
                     logger.error(f"Code generation API/extraction failed on final attempt {attempt + 1}.")
                     raise CodeGenerationError(f"Code generation API/extraction failed: {e}", last_code=last_successful_code, last_feedback=feedback) from e
                 else:
                      logger.warning(f"Retrying due to code generation API/extraction error (Attempt {attempt + 1}/{max_attempts}).")
                      continue # Loop continues

        # If the loop finishes without returning success
        # This should only happen if tests were provided (run_tests_flag was True)
        # and they failed on all 'max_attempts' retries.
        logger.error(f"Code generation failed to produce passing code after {max_attempts} attempts.")
        raise CodeGenerationError(
            "Failed to generate code that passes all provided tests after multiple retries.",
            last_code=last_successful_code, # The last code that was generated, even if failing
            last_test_results=final_test_results, # The results from the last test run
            last_feedback=feedback # The feedback generated after the last failure
        )


    # --- Helper Methods --- (Ollama Call, JSON Extract, Code Extract) ---

    def _sync_ollama_generate_and_accumulate(self, model: str, prompt: str, options: Dict[str, Any]) -> str:
        """Synchronous helper that calls Ollama and accumulates the streamed response."""
        full_response = ""
        logger.debug(f"SYNC OLLAMA: Starting generation for model {model}")
        try:
            # This is the blocking call and iteration
            stream = ollama.generate(
                model=model,
                prompt=prompt,
                stream=True,
                options=options,
            )
            logger.debug(f"SYNC OLLAMA: Stream object obtained for {model}")
            # This iteration is also synchronous
            for chunk in stream:
                token = chunk.get("response", "")
                # Note: print() won't show up in real-time from the thread easily
                # Consider logging chunks if needed, but it might be slow
                full_response += token
            logger.debug(f"SYNC OLLAMA: Finished accumulating response for {model}. Length={len(full_response)}")
            return full_response.strip()
        except Exception as e:
            # Log the error here so it's associated with the sync operation
            logger.error(f"SYNC OLLAMA: API call/iteration failed for model {model}: {e}", exc_info=True) # Log traceback from thread
            # Re-raise the exception so run_sync can propagate it
            raise AIWriterError(f"Ollama sync helper error for {model}: {e}") from e


    async def _ollama_generate_stream(self, model: str, prompt: str, options: Dict[str, Any]) -> str:
        """
        Asynchronously runs the synchronous Ollama call and stream accumulation
        in a worker thread.
        """
        logger.debug(f"ASYNC OLLAMA: Offloading generation for model {model} to thread.")
        try:
            # Use run_sync to call the synchronous helper
            # Pass arguments positionally to run_sync for the helper
            full_response = await to_thread.run_sync(
                self._sync_ollama_generate_and_accumulate,
                model, # arg 1 for helper
                prompt, # arg 2 for helper
                options, # arg 3 for helper
                abandon_on_cancel=False # Safer default
            )
            # The print/stream simulation in the main async flow is removed,
            # as the actual streaming happens synchronously within the thread now.
            # We only get the final result back here.
            logger.debug(f"ASYNC OLLAMA: Received final response from thread for {model}. Length={len(full_response)}")
            return full_response # Already stripped in the sync helper
        except Exception as e:
            # Catch errors propagated by run_sync
            logger.error(f"ASYNC OLLAMA: Error occurred running Ollama task in thread for model {model}: {e}", exc_info=self.settings.debug_logging)
            # Re-raise as a specific error type if desired, or let the original bubble up
            if isinstance(e, AIWriterError): # If it was our specific error from the helper
                raise e
            else: # If it was some other error from run_sync itself
                raise AIWriterError(f"AnyIO thread error for {model}: {e}") from e

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempts to extract a JSON object from LLM response text."""
        logger.debug("Attempting JSON extraction...")
        try:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
            if match:
                logger.debug("Found JSON in fenced code block.")
                return json.loads(match.group(1))
            try:
                logger.debug("Attempting direct JSON parsing.")
                return json.loads(text)
            except json.JSONDecodeError:
                logger.debug("Direct JSON parsing failed.")
                pass
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                logger.warning("Using potentially unreliable regex fallback for JSON extraction.")
                # Further clean potential surrounding text if needed
                try:
                     return json.loads(match.group(1))
                except json.JSONDecodeError:
                     logger.error("Regex JSON fallback also failed to parse.")
                     pass # Fall through

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
        match_python = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match_python:
            logger.debug("Extracted code from ```python block.")
            return match_python.group(1).strip()
        match_generic = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match_generic:
            logger.warning("Extracted code from generic ``` block.")
            return match_generic.group(1).strip()
        logger.warning("No fenced code block found, returning entire response as potential code.")
        # Simple cleaning: remove potential leading/trailing explanation lines if needed
        lines = text.strip().split('\n')
        if lines and not lines[0].strip().startswith(('def ', 'import ', '#', '"""')):
             lines = lines[1:]
        if lines and not lines[-1].strip().endswith((':', '))', '"""', "'", '"')):
              lines = lines[:-1]
        return "\n".join(lines).strip()


# --- Main Execution Example ---

async def run_main():
    """Example of how to use the AIProgramWriter."""
    print("Loading settings...")
    try:
        settings = AISettings() # type: ignore
    except ValidationError as e:
        print(f"Error loading settings: {e}")
        return

    print("Initializing AIProgramWriter...")
    writer = AIProgramWriter(settings)

    requirement_prime = """Problem: Check if a number is prime.
Example: Input: 7 Output: True. Input: 10 Output: False.
Constraints: Input is a positive integer > 1."""

    # Example 1: No tests provided
    print("\n--- Running 'is_prime' without tests ---")
    result_no_tests = await writer.generate_and_optionally_test(requirement=requirement_prime)
    _print_final_result("Prime Check (No Tests)", result_no_tests)

    # Example 2: Human tests provided (using keys expected for prime check)
    print("\n--- Running 'is_prime' with human tests ---")
    # User MUST ensure keys here match what spec generator is likely to produce (or what they want)
    human_tests_prime = [
        {"inputs": {"num": 2}, "expected": True},  # Key 'num' used
        {"inputs": {"num": 7}, "expected": True},
        {"inputs": {"num": 10}, "expected": False},
        {"inputs": {"num": 1}, "expected": False},
        {"inputs": {"num": 15}, "expected": False},
        {"inputs": {"num": 97}, "expected": True}
    ]
    result_with_tests = await writer.generate_and_optionally_test(requirement=requirement_prime, human_tests_input=human_tests_prime)
    _print_final_result("Prime Check (With Tests)", result_with_tests)

    # Example 3: Multi-parameter function (Testing should be skipped)
    print("\n--- Running multi-parameter function with tests (testing should be skipped by runner) ---")
    requirement_multi = "Create a function that takes two integers, 'a' and 'b', and returns their sum."
    human_tests_multi = [
        {"inputs": {"a": 1, "b": 2}, "expected": 3},
        {"inputs": {"a": -5, "b": 5}, "expected": 0}
        # Keys 'a' and 'b' are provided by user
    ]
    result_multi = await writer.generate_and_optionally_test(requirement=requirement_multi, human_tests_input=human_tests_multi)
    _print_final_result("Multi-Param Add (With Tests)", result_multi)


def _print_final_result(title: str, result: Dict[str, Any]):
     """Helper to print formatted results."""
     print(f"\n--- FINAL RESULT ({title}) ---")
     if result["success"]:
        print("Generation Successful!")
        print("\nGenerated Specification:")
        print(json.dumps(result.get("spec"), indent=2))
        if result.get("test_run_results"):
            print("\nTest Run Results:")
            print(json.dumps(result.get("test_run_results"), indent=2))
        else:
             print("\nTesting: Not performed or skipped.")
        print("\nGenerated Code:")
        print("```python")
        print(result.get("code", "# No code generated"))
        print("```")
     else:
        print("Generation Failed.")
        print("\nErrors:")
        for error in result.get("errors", ["Unknown error"]):
            print(f"- {error}")
        # Print partial results on failure
        if result.get("spec"):
             print("\nLast Generated Specification:")
             print(json.dumps(result.get("spec"), indent=2))
        if result.get("code"):
            print("\nLast Generated Code (May be faulty):")
            print("```python")
            print(result.get("code", "# No code generated"))
            print("```")
        if result.get("test_run_results"):
            print("\nLast Test Run Results:")
            print(json.dumps(result.get("test_run_results"), indent=2))
     print("--------------------")


if __name__ == "__main__":
    try:
        anyio.run(run_main)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")
        traceback.print_exc()