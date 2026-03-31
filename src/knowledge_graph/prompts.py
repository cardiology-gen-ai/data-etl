import json
from typing import List, Dict, Any


ENTITY_EXTRACTION_SINGLE_SYSTEM_PROMPT = """You are a medical terminology expert.

Extract normalized, reusable cardiology-related concepts.

Rules:
- Use lowercase
- Use singular form
- Prefer standard clinical terminology
- Do NOT include trivial words (e.g. management, approach, recommendations)
- Merge obvious synonyms when the expansion is explicit in the text
- Only return concepts that are explicitly supported by the text
- Assign exactly one type to each concept

Allowed types:
- disease
- phenotype
- diagnostic_test
- imaging_modality
- management
- risk_factor

Return JSON only.
"""


ENTITY_EXTRACTION_BATCH_SYSTEM_PROMPT = """You are a medical terminology expert.

For each provided section, extract normalized, reusable cardiology-related concepts.

Rules:
- Use lowercase
- Use singular form
- Prefer standard clinical terminology
- Do NOT include trivial words (e.g. management, approach, recommendations)
- Merge obvious synonyms when the expansion is explicit in the text
- Only return concepts that are explicitly supported by the text
- Assign exactly one type to each concept
- Keep concepts separated by section uid
- Return every provided uid exactly once, even if its concept list is empty

Allowed types:
- disease
- phenotype
- diagnostic_test
- imaging_modality
- management
- risk_factor

Return JSON only.

Expected format:
[
  {
    "uid": "section_uid_1",
    "concepts": [
      {"name": "hypertrophic cardiomyopathy", "type": "disease"}
    ]
  },
  {
    "uid": "section_uid_2",
    "concepts": []
  }
]
"""


def build_entity_extraction_single_user_prompt(text: str) -> str:
    return f'''
Text:
"""
{text}
"""

Return a JSON array of objects.

Example:
[
  {{"name": "hypertrophic cardiomyopathy", "type": "disease"}},
  {{"name": "sudden cardiac death", "type": "risk_factor"}}
]
'''.strip()


def build_entity_extraction_batch_user_prompt(
    sections_payload: List[Dict[str, Any]]
) -> str:
    return (
        "Sections:\n"
        + json.dumps(sections_payload, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON only."
    )