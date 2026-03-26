"""
Write Draft Plan Tool - Save a draft plan and run section-by-section evaluation
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool


# Mapping of checklist sections to their relevant tool names
SECTION_TOOL_MAPPING = {
    'intercity_and_accommodation': ['query_train_info', 'query_flight_info', 'query_hotel_info'],
    'meals': ['recommend_restaurants', 'query_restaurant_details'],
    'attractions': ['recommend_attractions', 'query_attraction_details'],
    'geo_intracity': ['search_location', 'query_road_route_info'],
}

EVAL_SYSTEM_PROMPT = """You are a meticulous auditor for a travel planning system. Your sole job is to verify whether a draft travel plan correctly satisfies a set of checklist requirements by cross-referencing against the provided source data (tool call results).

You have NO role in planning. You did not create this plan. You have no stake in it passing. Your job is to find errors.

## What You Receive

1. **User Query** — The original travel planning request with constraints and preferences.
2. **Tool Results** — Raw data returned by search/lookup tools (flights, hotels, restaurants, attractions, transport routes, etc.). This is the ground truth.
3. **Draft Plan** — The generated travel plan to validate.
4. **Checklist Items** — Specific requirements the plan must satisfy.

## Your Task

For each checklist item, determine whether the draft plan section satisfies it by verifying claims against the tool results. Do not assume correctness — verify every checkable fact.

## Verification Principles

- **Tool results are ground truth.** If the plan states a fact (price, duration, departure time, rating, cuisine type, distance, transfer time), it must match the tool results exactly. Any discrepancy is an error.
- **Assume nothing.** If a checklist item requires something and the plan does not explicitly address it, that is a failure — not an implicit pass.
- **Check arithmetic.** Verify all time calculations (arrival + duration = departure, transfer windows between activities), budget totals, and distance/routing logic by computing them yourself.
- **Check selection quality.** If the user expressed preferences (cuisine, budget range, rating threshold, accessibility, proximity), verify the plan selected options that best match — not just any valid option. Flag cases where a clearly better alternative exists in the tool results.
- **Check data transcription.** Verify that names, addresses, phone numbers, operating hours, prices, and other factual details copied from tool results into the plan are accurate. Character-level precision matters.
- **Check constraint satisfaction.** Verify hard constraints from the user query (dietary restrictions, budget caps, time windows, group size, mobility needs) are fully respected.
- **Be specific.** When reporting an error, state: what the plan says, what the tool results say, and why it is wrong. Cite exact values.

## Response Format

Respond with ONLY a JSON object — no markdown fences, no preamble, no commentary.

[
  {
    "verdict": "pass" | "fail",
    "finding": "<concise explanation — what was checked, what was found>",
    "plan_value": "<what the plan states, if relevant>",
    "source_value": "<what the tool results state, if relevant>",
  }
]"""


@register_tool('write_draft_plan')
class WriteDraftPlanTool(BaseTravelTool):
    """Tool for saving a draft plan and running evaluation against checklist"""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self._checklist = self._load_checklist()
        self._agent_context = cfg.get('agent_context', {}) if cfg else {}

    def _load_checklist(self) -> Dict[str, List[str]]:
        """Load the evaluation checklist from JSON file"""
        checklist_path = Path(__file__).parent / 'checklist.json'
        with open(checklist_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_user_prompt(self, messages: List[Dict]) -> str:
        """Extract the first user message from conversation history"""
        for msg in messages:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                return msg.get('content', '')
        return ''

    def _extract_tool_results(self, messages: List[Dict], tool_names: List[str]) -> str:
        """
        Extract stitched tool invocations and responses for given tool names.
        Each entry pairs the tool call (name + arguments) with its response.
        """
        # Build a map of tool_call_id -> (name, arguments) from assistant messages
        tool_call_map: Dict[str, tuple] = {}
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                for tc in msg['tool_calls']:
                    tc_id = tc.get('id', '')
                    func = tc.get('function', {})
                    name = func.get('name', '')
                    args = func.get('arguments', '')
                    if name in tool_names:
                        tool_call_map[tc_id] = (name, args)

        # Match tool responses by tool_call_id and stitch together
        results = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get('role') == 'tool':
                tc_id = msg.get('tool_call_id', '')
                if tc_id in tool_call_map:
                    name, args = tool_call_map[tc_id]
                    entry = (
                        f"Tool Call: {name}({args})\n"
                        f"Tool Response: {msg.get('content', '')}"
                    )
                    results.append(entry)

        return '\n\n'.join(results) if results else ''

    @staticmethod
    def _extract_token_usage(response) -> Optional[Dict[str, Any]]:
        """Extract token usage from an LLM response object."""
        usage = getattr(response, 'usage', None)
        if usage is None and isinstance(response, dict):
            usage = response.get('usage')
        if usage is None:
            return None

        def _read(field: str):
            if isinstance(usage, dict):
                return usage.get(field)
            return getattr(usage, field, None)

        prompt_tokens = _read('prompt_tokens')
        completion_tokens = _read('completion_tokens')
        total_tokens = _read('total_tokens')

        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return None

        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
        }

    @staticmethod
    def _parse_eval_response(raw: str) -> List[Dict]:
        """Parse the evaluation JSON array from LLM response text."""
        text = raw.strip()

        # Strip markdown code fences if present
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # Locate the outermost JSON array
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: try parsing the whole string
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Unparseable — surface as a single error item
        return [{
            "verdict": "error",
            "finding": f"Failed to parse evaluation response: {raw[:500]}",
        }]

    def _evaluate_section(
        self,
        section_name: str,
        checklist_items: List[str],
        user_prompt: str,
        draft_plan: str,
        tool_results: str,
        model: str,
    ) -> Dict:
        """Evaluate a single checklist section via LLM call.

        Returns:
            Dict with keys: section, raw_response, token_usage
        """
        from ..agent.call_llm import call_llm

        # Build user message
        parts = [
            f"## User Query\n{user_prompt}",
            f"## Draft Plan\n{draft_plan}",
        ]
        if tool_results:
            parts.append(f"## Relevant Tool Results\n{tool_results}")

        checklist_text = '\n'.join(f"- {item}" for item in checklist_items)
        parts.append(f"## Checklist for Section: {section_name}\n{checklist_text}")
        parts.append(
            "Evaluate the draft plan against each checklist item above. "
            "For each item, state whether it PASSES or FAILS and provide a brief explanation."
        )

        eval_messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": '\n\n'.join(parts)},
        ]

        response = call_llm(config_name=model, messages=eval_messages)
        content = response.choices[0].message.content or ''
        token_usage = self._extract_token_usage(response)

        return {
            'section': section_name,
            'raw_response': content,
            'token_usage': token_usage,
        }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Accept a draft plan, run section-by-section evaluation in parallel,
        and return only the failed checklist items grouped by section.

        Args:
            params: Dictionary containing:
                - draft_plan (str): Draft plan content

        Returns:
            JSON string — either {"message": "all checklist passed"} or
            a dict of section_name -> [failed items]
        """
        params = self._verify_json_format_args(params)
        draft_plan = params.get('draft_plan', '')

        messages = self._agent_context.get('messages', [])
        model = self._agent_context.get('model', '')

        user_prompt = self._extract_user_prompt(messages)

        # Build tasks for each section
        section_tasks = []
        for section_name, checklist_items in self._checklist.items():
            relevant_tools = SECTION_TOOL_MAPPING.get(section_name, [])
            tool_results = self._extract_tool_results(messages, relevant_tools) if relevant_tools else ''
            section_tasks.append((section_name, checklist_items, tool_results))

        # Evaluate all sections in parallel
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=len(section_tasks)) as executor:
            futures = {}
            for section_name, checklist_items, tool_results in section_tasks:
                future = executor.submit(
                    self._evaluate_section,
                    section_name, checklist_items,
                    user_prompt, draft_plan,
                    tool_results, model,
                )
                futures[future] = section_name

            for future in as_completed(futures):
                section = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'section': section,
                        'raw_response': '',
                        'token_usage': None,
                    })

        # Sort results back to original checklist section order
        section_order = list(self._checklist.keys())
        results.sort(
            key=lambda r: section_order.index(r['section'])
            if r['section'] in section_order else len(section_order)
        )

        # Aggregate token usage across all eval calls and store for the agent loop
        total_usage: Dict[str, int] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        per_section_usage: Dict[str, Any] = {}
        for r in results:
            tu = r.get('token_usage')
            if tu:
                per_section_usage[r['section']] = tu
                for key in total_usage:
                    if tu.get(key) is not None:
                        total_usage[key] += tu[key]

        self._agent_context['eval_token_usage'] = {
            'total': total_usage,
            'per_section': per_section_usage,
        }

        # Parse each section response, collect only failed items
        failures_by_section: Dict[str, List[Dict]] = {}
        for r in results:
            items = self._parse_eval_response(r.get('raw_response', ''))
            failed = [item for item in items if item.get('verdict') == 'fail']
            if failed:
                failures_by_section[r['section']] = failed

        if not failures_by_section:
            return json.dumps({"message": "all checklist passed"}, ensure_ascii=False)

        return json.dumps(failures_by_section, ensure_ascii=False)
