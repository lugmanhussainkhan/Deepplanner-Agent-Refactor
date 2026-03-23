"""
Fetch Checklist Tool - Retrieve checklist section items by optional section slug
"""
import os
from typing import Dict, List, Optional, Union
from xml.sax.saxutils import escape

from .base_travel_tool import BaseTravelTool, register_tool


@register_tool('fetch_checklist')
class FetchChecklistTool(BaseTravelTool):
    """Tool for fetching checklist section items and next section slug"""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.checklist_path = os.path.join(os.path.dirname(__file__), 'checklist.json')
        self.checklist_data = self.load_json_database(self.checklist_path)
        self.section_slugs: List[str] = list(self.checklist_data.keys())

    def _format_result_as_xml(
        self,
        section_slug: Optional[str] = None,
        instruction: str = "",
        checklist_items: Optional[List[str]] = None,
        next_section_slug: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        available_slugs: Optional[List[str]] = None
    ) -> str:
        checklist_items = checklist_items or []
        available_slugs = available_slugs or []

        lines: List[str] = ["<fetch_checklist_response>"]

        if error is not None:
            lines.append(f"  <error>{escape(error)}</error>")
            if available_slugs:
                lines.append("  <available_slugs>")
                for slug in available_slugs:
                    lines.append(f"    <slug>{escape(slug)}</slug>")
                lines.append("  </available_slugs>")
            lines.append("</fetch_checklist_response>")
            return "\n".join(lines)

        if section_slug is not None:
            lines.append(f"  <section_slug>{escape(section_slug)}</section_slug>")

        lines.append(f"  <instruction>{escape(instruction)}</instruction>")

        lines.append("  <checklist_items>")
        for item in checklist_items:
            lines.append(f"    <item>{escape(item)}</item>")
        lines.append("  </checklist_items>")

        if next_section_slug is None:
            lines.append("  <next_section_slug />")
        else:
            lines.append(f"  <next_section_slug>{escape(next_section_slug)}</next_section_slug>")

        if message is not None:
            lines.append(f"  <message>{escape(message)}</message>")

        lines.append("</fetch_checklist_response>")
        return "\n".join(lines)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Fetch checklist section content by optional section slug.

        Args:
            params: Optional dictionary containing:
                - section_slug (str): Checklist section slug/key

        Returns:
            XML string containing section checklist items and next section slug.
            For the last section, includes completion message.
            For invalid section_slug, returns error with available slugs.
        """
        params = self._verify_json_format_args(params)
        section_slug = params.get('section_slug')

        if not self.section_slugs:
            return self._format_result_as_xml(error="Checklist is empty.")

        if section_slug is None:
            current_index = 0
            current_slug = self.section_slugs[current_index]
        else:
            if section_slug not in self.checklist_data:
                return self._format_result_as_xml(
                    error=f"Invalid section_slug: '{section_slug}'",
                    available_slugs=self.section_slugs
                )
            current_slug = section_slug
            current_index = self.section_slugs.index(current_slug)

        next_index = current_index + 1
        next_section_slug = self.section_slugs[next_index] if next_index < len(self.section_slugs) else None

        section_data = self.checklist_data[current_slug]
        message = None
        if next_section_slug is None:
            message = "All checklist items completed, respond with the plan once the current checklist items are validated."

        return self._format_result_as_xml(
            section_slug=current_slug,
            instruction=section_data.get("instruction", ""),
            checklist_items=section_data.get("checklist", []),
            next_section_slug=next_section_slug,
            message=message
        )
