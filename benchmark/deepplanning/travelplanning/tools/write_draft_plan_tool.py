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

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Accept a draft plan string and return a success message.

        Args:
            params: Dictionary containing:
                - draft_plan (str): Draft plan content

        Returns:
            Fixed success message
        """
        params = self._verify_json_format_args(params)
        _ = params.get('draft_plan', '')
        return "Saved draft plan"
