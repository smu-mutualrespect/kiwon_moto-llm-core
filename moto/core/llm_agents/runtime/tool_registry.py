from __future__ import annotations


def get_available_tool_names() -> list[str]:
    return [
        "skills.load_skill_document",
        "schema.inspect_output_shape",
        "state.inspect_consistency",
        "mock_data.get_mock_template",
    ]
