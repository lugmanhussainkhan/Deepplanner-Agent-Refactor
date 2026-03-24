"""
Fetch Checklist Tool - Retrieve checklist section items by optional section slug
"""
import os
from typing import Dict, List, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool


@register_tool('fetch_checklist')
class FetchChecklistTool(BaseTravelTool):
    """Tool for fetching checklist section items and next section slug"""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.checklist_path = os.path.join(os.path.dirname(__file__), 'checklist.json')
        self.checklist_data = self.load_json_database(self.checklist_path)
        self.section_slugs: List[str] = list(self.checklist_data.keys())

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Fetch checklist section content by optional section slug.

        Args:
            params: Optional dictionary containing:
                - section_slug (str): Checklist section slug/key

        Returns:
            JSON string containing section checklist items and next section slug.
            For the last section, includes completion message.
            For invalid section_slug, returns error with available slugs.
        """
        params = self._verify_json_format_args(params)
        section_slug = params.get('section_slug')

        if not self.section_slugs:
            return self.format_result_as_json({
                "error": "Checklist is empty."
            })

        if section_slug is None:
            current_index = 0
            current_slug = self.section_slugs[current_index]
        else:
            if section_slug not in self.checklist_data:
                return self.format_result_as_json({
                    "error": f"Invalid section_slug: '{section_slug}'",
                    "available_slugs": self.section_slugs
                })
            current_slug = section_slug
            current_index = self.section_slugs.index(current_slug)

        next_index = current_index + 1
        next_section_slug = self.section_slugs[next_index] if next_index < len(self.section_slugs) else None
        
        response = {
            "section_slug": current_slug,
            "checklist_items": self.checklist_data[current_slug],
            "next_section_slug": next_section_slug
        }
        
        if section_slug is None:
            response["message"] = "Draft plan saved successfully. Validation Loop Initiated. Evaluate your draft against the checklist below. If corrections are needed, call 'write_draft_plan' again. If it passes, call 'fetch_checklist' with the next_section_slug."
        

        if next_section_slug is None:
            response["message"] = "All checklist items completed, respond with the plan once the current checklist items are validated."

        return self.format_result_as_json(response)