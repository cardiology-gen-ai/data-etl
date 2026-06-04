# src/knowledge_graph/recommendation_prompts.py

from __future__ import annotations

import json
from typing import Any


RECOMMENDATION_EXTRACTION_SYSTEM_PROMPT = """
You are an expert information extraction system for ESC clinical practice guidelines.

Your task is to extract explicit clinical recommendation statements from guideline text.

A recommendation statement is a sentence or table row that tells clinicians what should be done,
should be considered, may be considered, is recommended, is not recommended, is indicated,
is contraindicated, or should be avoided in a clinical scenario.

Extract recommendation statements from:
- formal recommendation tables;
- "What to do" / "What not to do" tables;
- explicit recommendation-like body text.

Do NOT extract:
- generic background explanations;
- epidemiology;
- definitions;
- pure evidence summaries without a clear clinical action;
- descriptions of study results unless they are phrased as a clinical recommendation;
- figure captions unless they contain an explicit recommendation.

Classify every extracted statement with statement_kind:
- "formal_recommendation": an explicit guideline recommendation, often with class/level.
- "patient_advice": advice clinicians should give patients or families.
- "practical_guidance": operational clinical guidance without formal class/level.
- "contextual_information": background/context needed to understand recommendations.
- "legal_or_local_rule": legal, regulatory, reimbursement, availability, or local protocol constraints.
- "factual_statement": descriptive facts, epidemiology, study findings, or definitions.
- "unclear": recommendation-like text where the kind cannot be determined.

Only formal_recommendation, patient_advice, and practical_guidance are actionable
recommendation candidates. Contextual, legal/local, and factual statements may be
returned only when they are easy to confuse with recommendations; mark them with
the correct statement_kind so deterministic review can route them separately.

Important rules:
- Return only JSON.
- Do not add explanations outside the JSON.
- Do not infer Class of Recommendation or Level of Evidence if they are not explicitly visible.
- If Class of Recommendation is not explicitly present, use null.
- If Level of Evidence is not explicitly present, use null.
- Use the exact source quote from the input text.
- The source_quote must be copied from the section text, not paraphrased.
- The text must be directly supported by the source_quote.
- Do not include recommends, not_recommends, or mentions items that are not
  explicitly present in the source_quote, except safe acronym expansion when the
  acronym appears in the source_quote.
- Preserve negation.
- Distinguish positive and negative recommendations.
- Link recommendations to provided clinical concepts only when the match is clear.
- If no recommendation is present, return an empty list.

Valid polarity values:
- "for"
- "against"
- "conditional"
- "unclear"

Valid source_type values:
- "recommendation_table"
- "recommendation_table_candidate"
- "what_to_do_table"
- "body_text"
- "key_message"
- "unclear"

Valid statement_kind values:
- "formal_recommendation"
- "patient_advice"
- "practical_guidance"
- "contextual_information"
- "legal_or_local_rule"
- "factual_statement"
- "unclear"

Valid class_of_recommendation values:
- "I"
- "IIa"
- "IIb"
- "III"
- null

Valid level_of_evidence values:
- "A"
- "B"
- "C"
- null

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

Field definitions:
- text: clean recommendation statement.
- source_quote: exact span from the input text supporting the extraction.
- polarity:
  - "for" means the action is recommended/indicated.
  - "against" means the action is not recommended, contraindicated, or should be avoided.
  - "conditional" means the action may be considered or should be considered in specific circumstances.
  - "unclear" means the statement is recommendation-like but polarity is difficult to determine.
- recommends: clinical actions, drugs, procedures, tests, strategies, or devices positively recommended.
- not_recommends: clinical actions, drugs, procedures, tests, strategies, or devices discouraged or not recommended.
- applies_to: patient group, condition, scenario, or clinical context to which the recommendation applies.
- conditioned_on: explicit condition, threshold, contraindication, risk factor, or prerequisite.
- mentions: other relevant clinical concepts mentioned in the recommendation.
- source_unit_kind/table_id/table_title/row_index: optional provenance fields when the source appears to be a table row or similar unit. Do not infer a full table structure.
- confidence: extraction confidence from 0.0 to 1.0.

If a recommendation mentions a clinical concept but the exact concept is not present in the provided concept list,
still include it in the textual arrays. Downstream code will decide whether it can be linked to an existing Concept node.

Examples:
- "ICD implantation is recommended in patients with..." -> formal_recommendation.
- "Patients should be counselled about medication effects..." -> patient_advice.
- "This drug is available only according to local reimbursement rules" -> legal_or_local_rule, not an actionable clinical recommendation.
- "Hypertrophic cardiomyopathy is a genetic disease" -> factual_statement, not an actionable clinical recommendation.
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

Section text:
\"\"\"
{section_text}
\"\"\"

Extract explicit recommendation statements from the section text according to the JSON schema.
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
