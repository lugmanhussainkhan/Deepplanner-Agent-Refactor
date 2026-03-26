"""
Custom Agent implementation - Framework-independent
Uses universal LLM calling for multiple providers
"""
import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    from .call_llm import call_llm
except ImportError:
    from call_llm import call_llm


class ToolsFnAgent:
    """
    Lightweight function-calling Agent (framework-independent):
    - Loads tool schemas from tools/tool_schema.json
    - Dynamically loads tool classes (BaseTravelTool subclasses)
    - Iteratively calls LLM and executes tool_calls until final answer
    """

    def __init__(self,
                 model: str,
                 sample_id: Optional[str] = None,
                 database_base_path: Optional[str] = None,
                 tool_schema_path: Optional[str] = None,
                 language: str = 'zh') -> None:
        """
        Initialize Agent
        
        Args:
            model: Model name (must exist in models_config.json)
            sample_id: Sample ID for database path resolution
            database_base_path: Base path to database directory
            tool_schema_path: Path to tool schema JSON file
            language: Language code ('zh' or 'en')
        """
        self._load_env_from_dotenv()
        
        self.model = model
        self.language = language
        
        default_schema = Path(__file__).resolve().parent.parent / 'tools' / f'tool_schema_{language}.json'
        self.tool_schema_path = tool_schema_path or str(default_schema)
        
        self.sample_id = sample_id
        if database_base_path:
            self.database_base_path = Path(database_base_path)
        else:
            project_root = Path(__file__).resolve().parent.parent
            self.database_base_path = project_root / 'database' / f'database_{language}'

        self._notes_store: list = []  # Shared notes list, reset per task
        self._checkpoints_store: list = []  # Shared checkpoints list, reset per task
        self._agent_context: dict = {'model': self.model, 'messages': []}  # Shared context for evaluation tools
        
        self.tools_schema = self._load_tool_schemas()
        self.openai_tools = self._build_openai_tools(self.tools_schema)
        self.tool_instances = self._load_tool_instances()
        
        if not Path(self.tool_schema_path).exists():
            raise FileNotFoundError(f"Tool schema not found: {self.tool_schema_path}")

    def _load_env_from_dotenv(self) -> None:
        """
        Load environment variables from .env file
        
        Searches for .env in the following order:
        1. Domain directory (travelplanning/)
        2. Project root (parent of domain)
        """
        try:
            # Try domain directory first
            domain_root = Path(__file__).resolve().parent.parent
            domain_dotenv = domain_root / '.env'
            
            # Try project root
            project_root = domain_root.parent
            project_dotenv = project_root / '.env'
            
            # Use project root .env if it exists, otherwise domain .env
            dotenv_path = project_dotenv if project_dotenv.exists() else domain_dotenv
            
            if not dotenv_path.exists():
                return
            
            for line in dotenv_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and (key not in os.environ):
                    os.environ[key] = val
        except Exception:
            pass

    def _load_tool_schemas(self) -> List[Dict[str, Any]]:
        """Load tool schemas from JSON file"""
        path = Path(self.tool_schema_path)
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and 'tools' in raw and isinstance(raw['tools'], list):
            return raw['tools']
        return [raw]

    def _build_openai_tools(self, schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build OpenAI tools format
        - If schema is already {type:function, function:{...}}, use as-is
        - Otherwise wrap as function definition
        """
        tools: List[Dict[str, Any]] = []
        
        enabled_custom_tools = os.getenv('ENABLED_CUSTOM_TOOLS', "").split(',')
        custom_tools = ["write_todo", "write_note", "get_notes", "write_draft_plan", "fetch_checklist", "create_checkpoint", "execute_code"]
        
        for s in schemas:
            if isinstance(s, dict) and s.get('type') == 'function' and isinstance(s.get('function'), dict):
                if s['function']['name'] in custom_tools and s['function']['name'] not in enabled_custom_tools:
                    continue
                
                tools.append(s)
                continue
            if not isinstance(s, dict):
                continue
            func = {
                "name": s.get('name'),
                "description": s.get('description', ''),
                "parameters": s.get('parameters', {}),
            }
            if func["name"]:
                tools.append({"type": "function", "function": func})
                
        print("=" * 80)
        print("Final tool schema:")
        print(json.dumps(tools, indent=2))
        print("=" * 80)
        
        return tools

    def _build_tool_config(self, tool_cls) -> Dict[str, Any]:
        """Build tool configuration with database path and language"""
        cfg = {
            'language': self.language  # Pass language to tool instance
        }
        
        # Inject shared stores for stateful tools
        tool_name = getattr(tool_cls, 'name', '')
        if tool_name in ('write_note', 'get_notes'):
            cfg['notes_store'] = self._notes_store
        if tool_name == 'create_checkpoint':
            cfg['checkpoints_store'] = self._checkpoints_store
        if tool_name == 'write_draft_plan':
            cfg['agent_context'] = self._agent_context
        
        if self.sample_id is None:
            return cfg
        
        sample_db_path = self.database_base_path / f'id_{self.sample_id}'
        
        db_mapping = {
            'query_train_info': 'trains/trains.csv',
            'query_flight_info': 'flights/flights.csv',
            'query_hotel_info': 'hotels/hotels.csv',
            'query_attraction_details': 'attractions/attractions.csv',
            'recommend_attractions': 'attractions/attractions.csv',
            'search_location': 'locations/locations_coords.csv',
            'query_road_route_info': 'transportation/distance_matrix.csv',
            'recommend_restaurants': 'restaurants/restaurants.csv',
            'query_restaurant_details': 'restaurants/restaurants.csv',
        }
        
        if tool_name in db_mapping:
            db_path = sample_db_path / db_mapping[tool_name]
            if db_path.exists():
                cfg['database_path'] = str(db_path)
        
        return cfg
    
    def _load_tool_instances(self) -> Dict[str, Any]:
        """Dynamically load tool instances"""
        instances: Dict[str, Any] = {}
        tools_dir = Path(__file__).resolve().parent.parent / 'tools'

        sys.path.insert(0, str(tools_dir.parent))
        sys.path.insert(0, str(tools_dir))

        try:
            import tools  # noqa: F401
        except Exception:
            return instances
        
        try:
            import importlib
            tools_mod = importlib.import_module('tools.base_travel_tool')
            base_tool_cls = getattr(tools_mod, 'BaseTravelTool', None)
        except Exception:
            return instances
        
        if base_tool_cls is None:
            return instances

        # First pass: load all tools except execute_code (which needs
        # references to the other tool instances).
        deferred_cls = None
        for cls in base_tool_cls.__subclasses__():
            tool_name = getattr(cls, 'name', '')
            if tool_name == 'execute_code':
                deferred_cls = cls
                continue
            try:
                tool_cfg = self._build_tool_config(cls)
                inst = cls(cfg=tool_cfg)
                inst_name = getattr(inst, 'name', None) or tool_name
                if inst_name:
                    instances[inst_name] = inst
            except Exception:
                continue

        # Second pass: initialise execute_code with access to all loaded tools.
        if deferred_cls is not None:
            try:
                cfg = self._build_tool_config(deferred_cls)
                cfg['tool_instances'] = instances
                inst = deferred_cls(cfg=cfg)
                inst_name = getattr(inst, 'name', None) or 'execute_code'
                instances[inst_name] = inst
            except Exception:
                pass
        
        return instances

    def _exec_tool(self, name: str, arguments_json: str) -> str:
        """Execute tool call"""
        inst = self.tool_instances.get(name)
        if not inst:
            return json.dumps({"error": f"tool '{name}' not found"}, ensure_ascii=False)
        
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except Exception:
            args = {}
        
        try:
            res = inst.call(args)
            return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _call_llm(self, messages: List[Any], tools: Optional[List[Dict[str, Any]]] = None):
        """Call LLM with unified handling for all models"""
        # Pass messages directly - OpenAI SDK can handle both dict and object formats
        return call_llm(
            config_name=self.model,
            messages=messages,
            tools=tools
        )

    def _detect_tool_calls(self, assistant_message) -> List[Dict[str, Any]]:
        """Detect and normalize tool calls"""
        import uuid
        
        tool_calls = getattr(assistant_message, 'tool_calls', None)
        calls: List[Dict[str, Any]] = []
        if not tool_calls:
            return calls
        
        for idx, tc in enumerate(tool_calls):
            try:
                # Generate unique ID if not provided by the model
                tool_call_id = tc.id
                if tool_call_id is None or not tool_call_id:
                    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                
                calls.append({
                    'id': tool_call_id,
                    'name': tc.function.name,
                    'arguments': tc.function.arguments,
                })
            except Exception:
                continue
        
        return calls

    def _format_checkpoints(self) -> str:
        """Format all accumulated checkpoints into a summary message."""
        parts = ["The following checkpoints summarize the progress made so far:\n"]
        for i, cp in enumerate(self._checkpoints_store, 1):
            parts.append(f"--- Checkpoint {i}: {cp['milestone_name']} ---")
            parts.append(f"Selections: {cp['exact_selections']}")
            parts.append(f"Next Steps: {cp['next_steps']}")
            parts.append("")
        return "\n".join(parts)

    def _apply_checkpoint_pruning(self, messages: List) -> List:
        """
        Prune messages for context management.

        Keeps only:
        1. The system prompt (if present)
        2. The first user message (original query)
        3. A new user message synthesised from all accumulated checkpoints
        """
        pruned: List[Dict[str, Any]] = []

        # Preserve system prompt
        for msg in messages:
            msg_dict = self._message_to_dict(msg) if not isinstance(msg, dict) else msg
            if msg_dict.get('role') == 'system':
                pruned.append(msg)
                break

        # Preserve first user message
        for msg in messages:
            msg_dict = self._message_to_dict(msg) if not isinstance(msg, dict) else msg
            if msg_dict.get('role') == 'user':
                pruned.append(msg)
                break

        # Add checkpoint summary as a new user message
        checkpoint_summary = self._format_checkpoints()
        pruned.append({"role": "user", "content": checkpoint_summary})

        return pruned
    
    def _extract_plan_content(self, text: str) -> str:
        """Extract content from <plan>...</plan> tags"""
        if not text:
            return ""
        
        # Remove <think>...</think> sections
        think_end_matches = list(re.finditer(r"</think>", text, flags=re.IGNORECASE))
        if think_end_matches:
            last_think_end = think_end_matches[-1]
            text = text[last_think_end.end():]
        
        # Extract <plan>...</plan>
        matches = re.findall(r"<plan>(.*?)</plan>", text, flags=re.DOTALL | re.IGNORECASE)
        if not matches:
            return ""
        
        cleaned = [m.strip() for m in matches if m.strip()]
        return "\n\n".join(cleaned) if cleaned else ""

    def _extract_token_usage(self, response) -> Optional[Dict[str, Any]]:
        """Extract token usage from LLM response object/dict."""
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

    def _message_to_dict(self, msg) -> Dict[str, Any]:
        """Convert message object to serializable dictionary"""
        if isinstance(msg, dict):
            return msg
        
        msg_dict: Dict[str, Any] = {}
        
        # Extract role
        if hasattr(msg, 'role'):
            msg_dict['role'] = msg.role
        elif hasattr(msg, 'get'):
            msg_dict['role'] = msg.get('role', 'assistant')
        else:
            msg_dict['role'] = 'assistant'
        
        # Extract content
        if hasattr(msg, 'content'):
            msg_dict['content'] = msg.content or ''
        elif isinstance(msg, dict) and 'content' in msg:
            msg_dict['content'] = msg['content'] or ''
        else:
            msg_dict['content'] = ''
        
        # Extract tool_calls if present
        tool_calls = getattr(msg, 'tool_calls', None)
        if tool_calls:
            calls_list = []
            for tc in tool_calls:
                try:
                    tool_call_id = getattr(tc, 'id', None) or ''
                    call_dict = {
                        'id': tool_call_id,
                        'type': 'function',
                        'function': {
                            'name': getattr(tc.function, 'name', '') if hasattr(tc, 'function') else '',
                            'arguments': getattr(tc.function, 'arguments', '') if hasattr(tc, 'function') else ''
                        }
                    }
                    calls_list.append(call_dict)
                except Exception:
                    continue
            if calls_list:
                msg_dict['tool_calls'] = calls_list
        
        # Preserve reasoning_content if present
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            msg_dict['reasoning_content'] = msg.reasoning_content
        
        return msg_dict

    def _serialize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert all messages in list to serializable dictionaries"""
        serialized = []
        for msg in messages:
            serialized.append(self._message_to_dict(msg))
        return serialized

    def run(self,
            user_query: str,
            system_prompt: Optional[str] = None,
            max_llm_calls: int = 100) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Agent main loop: Call LLM → Execute tools → Repeat until final answer
        
        Args:
            user_query: User query
            system_prompt: System prompt
            max_llm_calls: Maximum LLM calls
            
        Returns:
            (final_plan, messages): Final plan and complete message history
        """
        messages: List[Dict[str, Any]] = []
        full_messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            full_messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_query})
        full_messages.append({"role": "user", "content": user_query})
        
        # Share full_messages reference with evaluation tools (e.g. write_draft_plan)
        self._agent_context['messages'] = full_messages
        
        llm_budget = max_llm_calls
        
        while llm_budget > 0:
            llm_budget -= 1
            
            resp = self._call_llm(messages=messages, tools=self.openai_tools)
            
            msg = resp.choices[0].message
            calls = self._detect_tool_calls(msg)
            assistant_msg = self._message_to_dict(msg)
            token_usage = self._extract_token_usage(resp)
            if token_usage is not None:
                assistant_msg['token_usage'] = token_usage
            
            messages.append(assistant_msg)
            full_messages.append(assistant_msg)
            
            if calls:
                # Execute tool calls
                for call in calls:
                    tool_result = self._exec_tool(call['name'], call['arguments'])
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": call['id'],
                        "name": call['name'],
                        "content": tool_result,
                    }
                    # Attach eval token usage produced by write_draft_plan
                    if call['name'] == 'write_draft_plan':
                        eval_usage = self._agent_context.get('eval_token_usage')
                        if eval_usage:
                            tool_msg['eval_token_usage'] = eval_usage
                    messages.append(tool_msg)
                    full_messages.append(tool_msg)
                    
                if any(call['name'] == 'create_checkpoint' for call in calls):
                    messages = self._apply_checkpoint_pruning(messages)
                    
                continue
            
            # No tool calls → Return final answer
            # msg was already added to messages at line 343
            final_content = self._extract_plan_content(msg.content or '')
            return final_content, full_messages
        
        return "Reached max LLM calls without final answer.", full_messages


def run_agent_inference(
    model: str,
    language: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    output_dir: Path,
    workers: int = 10,
    max_llm_calls: int = 100,
    rerun_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run agent inference (batch processing)
    
    Args:
        model: Configuration name from models_config.json
        language: Language code ('zh' or 'en')
        test_data_path: Path to test data JSON file
        database_dir: Base path to database directory
        tool_schema_path: Path to tool schema JSON file
        output_dir: Output directory for results
        workers: Number of parallel workers
        max_llm_calls: Maximum LLM calls per sample
        rerun_ids: Optional list of specific IDs to rerun. If None, run all samples.
    
    Returns:
        Results summary dict
    """
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Filter samples if rerun_ids is specified
    if rerun_ids is not None:
        rerun_ids_set = set(str(id) for id in rerun_ids)  # Convert to strings for comparison
        original_count = len(test_data)
        test_data = [s for s in test_data if str(s.get('id')) in rerun_ids_set]
        print(f"  🔄 Filtered {original_count} samples to {len(test_data)} samples for rerun")
        
        if len(test_data) == 0:
            print(f"  ⚠️  Warning: No samples found matching the specified IDs")
            return {
                'total': 0,
                'success': 0,
                'failed': 0,
                'elapsed_time': 0,
                'results': []
            }
    
    test_data = test_data[11:30]
    
    # test_data_limit = os.getenv('TEST_DATA_LIMIT', None)
    # if test_data_limit is not None:
    #     test_data = test_data[:int(test_data_limit)]
    
    print(f"\n{'='*80}")
    print(f"Agent Inference")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Language: {language}")
    print(f"Samples: {len(test_data)}")
    print(f"Workers: {workers}")
    print(f"{'='*80}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'trajectories').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)
    
    try:
        from prompts import get_system_prompt
    except ImportError:
        from agent.prompts import get_system_prompt
    
    print_lock = Lock()
    results = []
    
    def process_sample(sample):
        sample_id_raw = sample.get('id', 'unknown')
        sample_id = f"id_{sample_id_raw}" if str(sample_id_raw).isdigit() else str(sample_id_raw)
        query = sample.get('query', '')
        
        try:
            with print_lock:
                print(f"\n🚀 Processing sample: {sample_id}")
            
            agent = ToolsFnAgent(
                model=model,
                sample_id=sample_id_raw,
                database_base_path=database_dir,
                language=language
            )
            
            system_prompt = get_system_prompt(language)
            start_time = time.time()
            
            final_plan, full_messages = agent.run(
                user_query=query,
                system_prompt=system_prompt,
                max_llm_calls=max_llm_calls
            )
            
            elapsed = time.time() - start_time
            
            # Ensure messages are serializable before writing
            serialized_messages = agent._serialize_messages(full_messages)
            
            result = {
                'id': sample_id,
                'query': query,
                'model': model,
                'language': language,
                'final_plan': final_plan,
                'messages': serialized_messages,  # Use serialized messages
                'elapsed_time': elapsed,
                'success': True,
            }
            
            trajectory_file = output_dir / 'trajectories' / f'{sample_id}.json'
            try:
                with open(trajectory_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            except TypeError as e:
                with print_lock:
                    print(f"⚠️  Sample {sample_id}: JSON serialization error: {e}")
                    print(f"   Attempting to identify problematic message...")
                # Try to identify which message causes the problem
                for i, msg in enumerate(serialized_messages):
                    try:
                        json.dumps(msg, ensure_ascii=False)
                    except TypeError as msg_err:
                        with print_lock:
                            print(f"   Message {i} cannot be serialized: {msg_err}")
                            print(f"   Message type: {type(msg)}")
                            if hasattr(msg, '__dict__'):
                                print(f"   Message attrs: {list(msg.__dict__.keys())}")
                raise
            
            if final_plan:
                plan_file = output_dir / 'reports' / f'{sample_id}.txt'
                with open(plan_file, 'w', encoding='utf-8') as f:
                    f.write(final_plan)
            else:
                with print_lock:
                    print(f"⚠️  Sample {sample_id}: No plan extracted")
            
            with print_lock:
                print(f"✅ Sample {sample_id} completed in {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            with print_lock:
                print(f"❌ Sample {sample_id} failed: {e}")
            
            return {
                'id': sample_id,
                'query': query,
                'success': False,
                'error': str(e),
            }
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in test_data]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    success_count = sum(1 for r in results if r['success'])
    
    return {
        'total': len(results),
        'success': success_count,
        'failed': len(results) - success_count,
        'results': results
    }


if __name__ == '__main__':
    
    """Simple test"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen-plus', help='Configuration name from models_config.json')
    parser.add_argument('--language', default='zh', help='Language: zh or en')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    test_output_dir = base_dir / 'results' / 'test'
    
    result = run_agent_inference(
        model=args.model,
        language=args.language,
        test_data_path=base_dir / 'data' / f'travelplanning_query_{args.language}.json',
        database_dir=base_dir / 'database' / f'database_{args.language}',
        tool_schema_path=base_dir / 'tools' / f'tool_schema_{args.language}.json',
        output_dir=test_output_dir,
        workers=2,
    )
    print(f"\nTest completed: {result['success']}/{result['total']} succeeded")
