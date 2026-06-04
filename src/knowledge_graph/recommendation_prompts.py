# src/knowledge_graph/recommendation_prompts.py

from __future__ import annotations

import json
from typing import Any


RECOMMENDATION_EXTRACTION_SYSTEM_PROMPT = """
You are an expert information extraction system for ESC clinical practice guidelines.

Extract guideline-aware recommendation records from the input text.

Be conservative. Extract a record only when the text explicitly states what clinicians, patients, families, or healthcare services should do, consider, avoid, offer, discuss, monitor, test, treat, or not do.

Do NOT extract generic background, definitions, epidemiology, prognosis, study results, descriptive facts, reassuring statements, or statements of ability/eligibility unless they contain an explicit action.

Allowed statement_kind values:
- formal_recommendation: formal ESC/guideline recommendation, usually from a Recommendation Table or "What to do / What not to do" section, often with Class/Level.
- patient_advice: advice clinicians should communicate to patients/families about lifestyle, counselling, medication, work, school, driving, reproduction, or daily activity.
- practical_guidance: operational guidance about referral, specialist discussion, follow-up, service provision, care organization, or implementation.
- contextual_information: background/reassuring/explanatory information, not an actionable recommendation.
- legal_or_local_rule: legal, licensing, reimbursement, availability, or local-policy constraint.
- factual_statement: definition, epidemiology, disease description, or study finding.
- unclear: recommendation-like text whose type is uncertain.

Only formal_recommendation, patient_advice, and practical_guidance are actionable candidates.
Use contextual_information, legal_or_local_rule, or factual_statement only for text that may be confused with a recommendation but should be routed away from actionable recommendations.

Rules:
- Return only valid JSON.
- Use only information explicitly present in the input.
- Do not infer Class of Recommendation or Level of Evidence; use null unless explicitly visible.
- Preserve negation.
- source_quote must be copied exactly from the input text, not paraphrased.
- text must be directly supported by source_quote.
- For table rows, source_quote should include the minimal full row/span needed to support the action, population, condition, and negation.
- Do not add recommends, not_recommends, applies_to, conditioned_on, or mentions items unless they are explicitly supported by source_quote.
- Safe acronym expansion is allowed only when the acronym appears in source_quote.
- If no useful recommendation-like content is present, return {"recommendations": []}.

Source type:
- recommendation_table: formal labelled Recommendation Table.
- recommendation_table_candidate: table-like guidance that is not clearly a formal Recommendation Table.
- what_to_do_table: explicit "What to do" / "What not to do" section.
- body_text: narrative recommendation-like prose.
- key_message: explicit key-message section.
- unclear: source cannot be determined.

Examples:
- "It is recommended that all patients with cardiomyopathy and their relatives have access to multidisciplinary teams... I C"
  -> formal_recommendation; polarity=for; class=I; level=C.
- "Patients should be encouraged to maintain a recommended body mass index."
  -> patient_advice; polarity=for; recommends=["maintain a recommended body mass index"].
- "Avoid dehydration, excess alcohol intake, and drugs consumption."
  -> patient_advice; polarity=against; not_recommends=["dehydration", "excess alcohol intake", "drugs consumption"].
- "The implications of heavily manual jobs that involve strenuous activity should be discussed with the appropriate specialist."
  -> practical_guidance; polarity=for.
- "Most people with cardiomyopathy will be able to accomplish their normal jobs."
  -> contextual_information; recommends=[].
- "Advice on driving licences for heavy goods or passenger-carrying vehicles should be in line with local legislation."
  -> legal_or_local_rule; recommends=[].

Output schema:
{
  "recommendations": [
    {
      "text": string,
      "source_quote": string,
      "polarity": "for" | "against" | "conditional" | "unclear",
      "class_of_recommendation": "I" | "IIa" | "IIb" | "III" | null,
      "level_of_evidence": "A" | "B" | "C" | null,
      "source_type": "recommendation_table" | "recommendation_table_candidate" | "what_to_do_table" | "body_text" | "key_message" | "unclear",
      "statement_kind": "formal_recommendation" | "patient_advice" | "practical_guidance" | "contextual_information" | "legal_or_local_rule" | "factual_statement" | "unclear",
      "source_unit_kind": string | null,
      "table_id": string | null,
      "table_title": string | null,
      "row_index": number | null,
      "recommends": [string],
      "not_recommends": [string],
      "applies_to": [string],
      "conditioned_on": [string],
      "mentions": [string],
      "confidence": number
    }
  ]
}
""".strip()


RECOMMENDATION_EXTRACTION_USER_TEMPLATE = """
Document ID: {doc_id}
Section ID: {section_id}
Section title: {section_title}
Page start: {page_start}
Page end: {page_end}
Source unit kind: {source_unit_kind}
Source unit index: {source_unit_index}

Clinical concepts already extracted from this section:
{concepts_json}

Input text:
\"\"\"
{section_text}
\"\"\"

Extract recommendation-like records according to the JSON schema.
Be conservative: actionable records must contain an explicit action. Route descriptive, factual, contextual, or legal/local statements to the appropriate non-actionable statement_kind.
{recommendation_limit_instruction}
Return only valid JSON.
""".strip()


def build_recommendation_extraction_prompt(
    *,
    doc_id: str,
    section_id: str,
    section_title: str | None,
    page_start: int | None,
    page_end: int | None,
    section_text: str,
    concepts: list[dict[str, Any]] | None = None,
    max_recommendations: int | None = None,
    source_unit_kind: str | None = None,
    source_unit_index: int | None = None,
) -> str:
    """Build the user prompt for recommendation extraction.

    `concepts` should contain already extracted Concept nodes from the same section.
    Expected fields are flexible, but useful keys are:
    - name
    - normalized_name
    - canonical_type
    """

    concepts_json = json.dumps(
        concepts or [],
        ensure_ascii=False,
        indent=2,
    )

    recommendation_limit_instruction = ""
    if max_recommendations is not None:
        recommendation_limit_instruction = (
            f"Extract at most {int(max_recommendations)} recommendation records "
            "from this text. Prefer formal recommendation table rows and clearly "
            "actionable statements. If more than "
            f"{int(max_recommendations)} are present, return only the first "
            f"{int(max_recommendations)} in source order."
        )

    return RECOMMENDATION_EXTRACTION_USER_TEMPLATE.format(
        doc_id=doc_id,
        section_id=section_id,
        section_title=section_title or "",
        page_start=page_start if page_start is not None else "",
        page_end=page_end if page_end is not None else "",
        source_unit_kind=source_unit_kind or "",
        source_unit_index=(
            source_unit_index if source_unit_index is not None else ""
        ),
        concepts_json=concepts_json,
        section_text=section_text,
        recommendation_limit_instruction=recommendation_limit_instruction,
    )


def build_recommendation_messages(
    *,
    doc_id: str,
    section_id: str,
    section_title: str | None,
    page_start: int | None,
    page_end: int | None,
    section_text: str,
    concepts: list[dict[str, Any]] | None = None,
    max_recommendations: int | None = None,
    source_unit_kind: str | None = None,
    source_unit_index: int | None = None,
) -> list[dict[str, str]]:
    """Build chat-style messages for the LLM client."""

    return [
        {
            "role": "system",
            "content": RECOMMENDATION_EXTRACTION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_recommendation_extraction_prompt(
                doc_id=doc_id,
                section_id=section_id,
                section_title=section_title,
                page_start=page_start,
                page_end=page_end,
                section_text=section_text,
                concepts=concepts,
                max_recommendations=max_recommendations,
                source_unit_kind=source_unit_kind,
                source_unit_index=source_unit_index,
            ),
        },
    ]