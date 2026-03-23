"""
Write Draft Plan Tool - Save a draft plan string
"""
from typing import Dict, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool


@register_tool('write_draft_plan')
class WriteDraftPlanTool(BaseTravelTool):
    """Tool for saving a draft plan string"""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self._first_call = True

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Accept a draft plan string and return a success message.

        On first call, also fetches and returns the first section checklist
        to optimize the workflow (saves one turn).

        Args:
            params: Dictionary containing:
                - draft_plan (str): Draft plan content

        Returns:
            On first call: First section checklist from fetch_checklist
            On subsequent calls: Success message
        """
        params = self._verify_json_format_args(params)
        _ = params.get('draft_plan', '')

        if self._first_call:
            self._first_call = False
            # Import here to avoid circular imports
            from .fetch_checklist_tool import FetchChecklistTool
            fetch_tool = FetchChecklistTool(self.cfg)
            return fetch_tool.call({})

        return "Draft plan saved successfully."
