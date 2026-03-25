"""
Execute Code Tool - Run Python code with access to travel planning tool functions
"""
import io
import sys
import json
import traceback
from typing import Dict, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool


# The querying tools that should be exposed in the code execution environment
QUERYABLE_TOOL_NAMES = [
    'query_train_info',
    'query_flight_info',
    'query_hotel_info',
    'query_road_route_info',
    'query_attraction_details',
    'query_restaurant_details',
    'recommend_attractions',
    'recommend_restaurants',
    'search_location',
]


def _make_tool_wrapper(tool_instance):
    """
    Create a callable wrapper around a tool instance.

    The wrapper accepts keyword arguments, forwards them as a dict
    to ``tool_instance.call(params)``, and returns the parsed result
    (JSON-decoded if possible, otherwise the raw string).
    """

    def wrapper(**kwargs):
        result_str = tool_instance.call(kwargs)
        # Try to return structured data when the tool outputs JSON
        try:
            return json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            return result_str

    # Preserve the tool name for discoverability inside the sandbox
    wrapper.__name__ = getattr(tool_instance, 'name', 'unknown_tool')
    wrapper.__doc__ = getattr(tool_instance, 'description', '')
    return wrapper


@register_tool('execute_code')
class ExecuteCodeTool(BaseTravelTool):
    """
    Tool that executes Python code with travel planning tool functions
    available in the execution namespace.

    The following functions are injected into the execution environment:
        query_train_info, query_flight_info, query_hotel_info,
        query_road_route_info, query_attraction_details,
        query_restaurant_details, recommend_attractions,
        recommend_restaurants, search_location

    Each function accepts keyword arguments matching the tool's parameter
    schema and returns the parsed result.
    """

    # Maximum output length to avoid overwhelming the LLM context
    MAX_OUTPUT_LENGTH = 50000
    # Execution timeout (seconds) – kept generous for CSV-heavy queries
    EXEC_TIMEOUT_SECONDS = 120

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        # tool_instances will be injected by the agent via cfg
        self._tool_instances: Dict[str, object] = cfg.get('tool_instances', {}) if cfg else {}

        # Persistent state across calls: accumulated code blocks and a
        # shared namespace so that variables / functions defined in earlier
        # turns remain available in later ones.
        self._code_blocks: list[str] = []
        self._persistent_namespace: Optional[Dict] = None

    def _build_namespace(self) -> Dict:
        """Build the execution namespace with tool wrapper functions and common imports."""
        namespace: Dict = {'__builtins__': __builtins__}

        # Inject standard-library modules commonly useful for data wrangling
        import math, datetime, collections, itertools, functools, re  # noqa: E401
        namespace.update({
            'json': json,
            'math': math,
            'datetime': datetime,
            're': re,
            'collections': collections,
            'itertools': itertools,
            'functools': functools,
        })

        # Try to inject pandas (already loaded by the CSV tools)
        try:
            import pandas as pd
            namespace['pd'] = pd
            namespace['pandas'] = pd
        except ImportError:
            pass

        # Wrap each queryable tool and inject it as a top-level function
        for tool_name in QUERYABLE_TOOL_NAMES:
            inst = self._tool_instances.get(tool_name)
            if inst is not None:
                namespace[tool_name] = _make_tool_wrapper(inst)

        return namespace

    def _get_or_create_namespace(self) -> Dict:
        """Return the persistent namespace, creating it on first use."""
        if self._persistent_namespace is None:
            self._persistent_namespace = self._build_namespace()
        return self._persistent_namespace

    def reset(self) -> None:
        """Clear accumulated code blocks and the persistent namespace."""
        self._code_blocks.clear()
        self._persistent_namespace = None

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute the provided Python code.

        Code blocks are **accumulated** across calls so that variables,
        functions, and imports defined in earlier turns remain available in
        later ones.  The full accumulated code is re-executed each time to
        ensure a consistent environment, but only the **new** block's stdout
        and return value are reported back.

        Args:
            params: Must contain a ``code`` key with the Python source code.
                    May optionally contain ``reset`` (bool, default False).
                    When *reset* is True the accumulated history is cleared
                    and execution starts fresh from the current code block.

        Returns:
            A string combining captured stdout and the value of the last
            expression (if any), or an error traceback on failure.
        """
        params = self._verify_json_format_args(params)
        code = params.get('code', '')
        should_reset = params.get('reset', False)

        # Normalise truthy string values ("true", "True", "1") coming from
        # JSON-parsed LLM output.
        if isinstance(should_reset, str):
            should_reset = should_reset.lower() in ('true', '1', 'yes')

        if not code or not code.strip():
            return json.dumps({'error': 'No code provided'}, ensure_ascii=False)

        # ── Handle reset ──────────────────────────────────────────────
        if should_reset:
            self.reset()

        # ── Accumulate the new code block ─────────────────────────────
        self._code_blocks.append(code)

        # Build the full source: all prior blocks executed silently, then
        # the new block whose output we actually capture.
        prior_code = '\n'.join(self._code_blocks[:-1])
        new_code = self._code_blocks[-1]

        namespace = self._get_or_create_namespace()

        # ── Re-run prior blocks silently to rebuild state ─────────────
        if prior_code.strip():
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()  # discard prior output
            try:
                exec(compile(prior_code, '<execute_code:history>', 'exec'), namespace)
            except Exception:
                # Prior blocks previously succeeded; if they now fail it
                # likely means the new block hasn't been executed yet, so
                # we still continue and let the user see the real error.
                pass
            finally:
                sys.stdout = old_stdout

        # ── Execute the new code block and capture output ─────────────
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        result_value = None
        error_output = None

        try:
            import ast
            tree = ast.parse(new_code)

            # If the last statement is an expression, pop it and eval
            # separately so we can capture its value (REPL-style).
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
                ast.fix_missing_locations(last_expr)

            if tree.body:
                exec(compile(tree, '<execute_code>', 'exec'), namespace)

            if last_expr is not None:
                result_value = eval(compile(last_expr, '<execute_code>', 'eval'), namespace)

        except Exception:
            error_output = traceback.format_exc()
            # Roll back the failed block so it doesn't poison future turns
            self._code_blocks.pop()
        finally:
            sys.stdout = old_stdout

        # ── Assemble output ───────────────────────────────────────────
        stdout_text = captured_output.getvalue()
        parts = []

        if stdout_text:
            parts.append(stdout_text.rstrip())

        if error_output:
            parts.append(f"[ERROR]\n{error_output}")
        elif result_value is not None:
            try:
                formatted = json.dumps(result_value, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                formatted = repr(result_value)
            parts.append(formatted)

        output = '\n'.join(parts) if parts else '(no output)'

        if len(output) > self.MAX_OUTPUT_LENGTH:
            output = output[:self.MAX_OUTPUT_LENGTH] + f'\n... [truncated, total {len(output)} chars]'

        return output
