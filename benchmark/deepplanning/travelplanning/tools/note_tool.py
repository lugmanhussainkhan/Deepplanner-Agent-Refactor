"""
Note Tools - Write and retrieve notes during travel planning (task-scoped)
"""
from typing import Dict, List, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool


@register_tool('write_note')
class WriteNoteTool(BaseTravelTool):
    """Tool for writing a note to the task-scoped notes list"""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        # Shared mutable list reference injected via cfg by the agent
        self._notes: List[str] = cfg.get('notes_store', []) if cfg else []

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Append a note to the notes list.

        Args:
            params: Dictionary containing 'note' (str)

        Returns:
            Confirmation string
        """
        params = self._verify_json_format_args(params)
        note: str = params.get('note', '')
        if not note:
            return "Note content cannot be empty"
        self._notes.append(note)
        return f"Note saved. Total notes: {len(self._notes)}"


@register_tool('get_notes')
class GetNotesTool(BaseTravelTool):
    """Tool for retrieving all notes from the task-scoped notes list"""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        # Same shared mutable list reference
        self._notes: List[str] = cfg.get('notes_store', []) if cfg else []

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Return all notes.

        Returns:
            JSON-encoded list of note strings
        """
        return self.format_result_as_json(self._notes)
