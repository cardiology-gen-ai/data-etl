import json
from typing import List, Dict, Any


ENTITY_EXTRACTION_SINGLE_SYSTEM_PROMPT = """You are a medical terminology expert working on cardiology guidelines.

Task:
Extract normalized, reusable cardiology-related concepts from the provided text.

General rules:
- Use lowercase
- Use singular form when appropriate
- Prefer standard clinical terminology
- Only extract concepts that are explicitly supported by the text
- Do NOT invent concepts that are only implied
- Do NOT include overly generic or trivial words such as:
  management, treatment, therapy, recommendation, patient, disease, risk, test, procedure, care
- Merge obvious synonyms only when the equivalence is explicit in the text
- Each extracted concept must have exactly one canonical type chosen from the allowed types below
- Return JSON only
- Do not add explanations or commentary

Allowed canonical types:
- disease
- clinical_finding
- risk_factor
- genetic_factor
- biomarker
- diagnostic_test
- imaging_modality
- score_or_risk_model
- drug_or_drug_class
- procedure_or_intervention
- device
- complication_or_comorbidity
- care_strategy
- anatomical_structure
"""


ENTITY_EXTRACTION_BATCH_SYSTEM_PROMPT = """You are a medical terminology expert working on cardiology guidelines.

Task:
For each provided section, extract normalized, reusable cardiology-related concepts.

General rules:
- Use lowercase
- Use singular form when appropriate
- Prefer standard clinical terminology
- Only extract concepts that are explicitly supported by the text
- Do NOT invent concepts that are only implied
- Do NOT include overly generic or trivial words such as:
  management, treatment, therapy, recommendation, patient, disease, risk, test, procedure, care
- Merge obvious synonyms only when the equivalence is explicit in the text
- Each extracted concept must have exactly one canonical type chosen from the allowed types below
- Keep concepts separated by section uid
- Return every provided uid exactly once, even if its concept list is empty
- Return JSON only
- Do not add explanations or commentary

Allowed canonical types:
- disease
- clinical_finding
- risk_factor
- genetic_factor
- biomarker
- diagnostic_test
- imaging_modality
- score_or_risk_model
- drug_or_drug_class
- procedure_or_intervention
- device
- complication_or_comorbidity
- care_strategy
- anatomical_structure

Expected format:
[
  {
    "uid": "section_uid_1",
    "concepts": [
      {"name": "hypertrophic cardiomyopathy", "type": "disease"},
      {"name": "left ventricular outflow tract obstruction", "type": "clinical_finding"}
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

Return a JSON array of objects in this format:
[
  {{"name": "hypertrophic cardiomyopathy", "type": "disease"}},
  {{"name": "late gadolinium enhancement", "type": "biomarker"}},
  {{"name": "echocardiography", "type": "imaging_modality"}}
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