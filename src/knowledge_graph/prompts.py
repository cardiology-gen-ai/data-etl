import json
from typing import List, Dict, Any


ENTITY_TYPE_GUIDANCE = """Type guidance:
- disease: a named disease entity, syndrome, or disorder.
- clinical_finding: a symptom, sign, phenotype, ECG/imaging/laboratory finding, or other clinical observation/result; not a disease.
- risk_factor: a variable, trait, exposure, or condition associated with increased risk.
- genetic_factor: a gene, mutation, variant, genotype, or inherited molecular cause.
- biomarker: a measurable biological or laboratory marker/analyte; not the test used to measure it.
- diagnostic_test: a test, examination, investigation, or assessment performed on the patient.
- imaging_modality: an imaging technology or modality in general.
- score_or_risk_model: a named score, calculator, prediction rule, or risk model.
- drug_or_drug_class: a drug, medication, or named drug class.
- procedure_or_intervention: a therapeutic, invasive, surgical, catheter-based, or electrophysiology intervention.
- device: a medical device, prosthesis, lead, valve system, monitor, or implant.
- complication_or_comorbidity: a complication, consequence, associated condition, or coexisting disease.
- care_strategy: a broader management, follow-up, screening, counselling, prevention, or surveillance strategy.
- anatomical_structure: an anatomical body structure.
- clinical_outcome: a clinically meaningful outcome, endpoint, event, or prognostic consequence, such as mortality, cardiovascular death, hospitalization, stroke, bleeding event, or symptom improvement.
"""


ENTITY_DISAMBIGUATION_GUIDANCE = """Disambiguation guidance:
- When a term could be either a diagnostic_test or an imaging_modality, classify it based on how it is used in the local text context.
- Use diagnostic_test for performed examinations such as genetic testing, echocardiography, exercise testing, or coronary computed tomography angiography.
- Use imaging_modality for general imaging technologies such as echocardiography, computed tomography, cardiac magnetic resonance, or nuclear imaging when referenced as modalities.
- Use clinical_finding for findings/results such as late gadolinium enhancement, T wave inversion, left ventricular hypertrophy, reduced ejection fraction, chest pain, syncope, or reduced ejection fraction.
- Use biomarker for markers such as troponin, BNP, NT-proBNP, creatinine, or C-reactive protein.
- Use disease for named disease entities such as hypertrophic cardiomyopathy, heart failure, amyloidosis, myocarditis, aortic stenosis, or atrial fibrillation.
- Use anatomical_structure only for body structures such as left ventricle, mitral valve, interventricular septum, coronary artery, or aortic valve.
- Do not label a finding as anatomical_structure when the full phrase expresses a pathological or clinical state. For example, left ventricular hypertrophy should be clinical_finding, not anatomical_structure.
- Do not label a disease, finding, or procedure as care_strategy just because it appears in a management section.
- Use clinical_outcome for named or specific outcomes/endpoints such as all-cause mortality, cardiovascular death, heart failure hospitalization, major bleeding, stroke, or symptom improvement. Do not extract generic words such as outcome, endpoint, event, or prognosis by themselves.
"""


ENTITY_EXTRACTION_GENERAL_RULES = """General rules:
- Use lowercase.
- Use singular form when appropriate.
- Prefer standard clinical terminology.
- Only extract concepts that are explicitly supported by the text.
- Do NOT invent concepts that are only implied.
- Do NOT return duplicate concepts.
- Prefer the most specific clinically meaningful span explicitly present in the text; do not split a single concept into smaller fragments if the longer expression is the real concept.
- Do NOT extract generic parent concepts from a specific concept unless the generic concept is separately and explicitly meaningful in the text.
- Do NOT include overly generic or trivial words such as:
  management, treatment, therapy, recommendation, patient, disease, risk, test, procedure, care.
- Do NOT extract broad organizational headings such as diagnosis, management, follow-up, screening, prognosis, risk stratification, or therapy unless the phrase refers to a specific reusable clinical concept.
- Do NOT extract section themes or broad topic labels unless they are themselves clinically meaningful concepts.
- Do NOT extract actions or abstract care verbs such as evaluate, consider, recommend, screen, monitor, or assess unless they appear as part of a named reusable concept.
- Do NOT extract recommendation labels or evidence labels alone, such as Class I, Class IIa, Class IIb, Class III, Level A, Level B, or Level C.
- Do NOT extract standalone adjectives, severity words, or modifiers such as mild, moderate, severe, symptomatic, asymptomatic, primary, secondary, advanced, acute, chronic, recurrent, persistent, isolated, or familial.
- Keep adjectives or modifiers only when they are part of a specific clinically meaningful concept, for example severe aortic stenosis or familial hypercholesterolaemia.
- Merge obvious synonyms only when the equivalence is explicit in the text.
- Do NOT expand abbreviations or acronyms unless the full form or equivalence is explicit in the text.
- If only an abbreviation or acronym appears in the text and it is clinically meaningful, extract the abbreviation itself in lowercase rather than guessing the expansion.
- If both the acronym and the full form are explicitly provided in the text, prefer the full form as the concept name.
- Be conservative with very short acronyms that become common English words when lowercased, such as AS or MR; extract them only if the source clearly uses them as medical abbreviations.
- Each extracted concept must have exactly one canonical type chosen from the allowed types below.
- Return valid JSON only.
- Use double quotes in JSON.
- Do not add explanations, commentary, markdown fences, or extra text before or after the JSON.
"""


ENTITY_EXTRACTION_EXAMPLES = """Examples:

Example 1:
Input text:
\"\"\"
Title: Diagnosis
\"\"\"

Correct output:
[]

Do NOT output:
[
  {"name": "diagnosis", "type": "diagnostic_test"},
  {"name": "diagnosis", "type": "care_strategy"}
]

Reason:
- diagnosis is only a broad section heading.
- a title alone should not create generic concepts unless the title itself is a specific clinical concept.


Example 2:
Input text:
\"\"\"
Title: Definitions

Body:
Acute coronary syndrome (ACS) may present with chest pain, changes on a 12-lead electrocardiogram, and elevated cardiac troponin. A final diagnosis may be acute myocardial infarction or unstable angina.
\"\"\"

Correct output:
[
  {"name": "acute coronary syndrome", "type": "disease"},
  {"name": "chest pain", "type": "clinical_finding"},
  {"name": "12-lead electrocardiogram", "type": "diagnostic_test"},
  {"name": "cardiac troponin", "type": "biomarker"},
  {"name": "acute myocardial infarction", "type": "disease"},
  {"name": "unstable angina", "type": "disease"}
]

Do NOT output:
[
  {"name": "acs", "type": "disease"},
  {"name": "changes", "type": "clinical_finding"},
  {"name": "elevated", "type": "clinical_finding"},
  {"name": "diagnosis", "type": "care_strategy"},
  {"name": "patient", "type": "clinical_finding"}
]

Reason:
- acute coronary syndrome is preferred over ACS because the full form is explicit in the text.
- changes and elevated are too generic by themselves.
- diagnosis and patient are generic.


Example 3:
Input text:
\"\"\"
Title: Imaging

Body:
CMR showed LGE in the left ventricle. No full acronym definitions are provided in this section.
\"\"\"

Correct output:
[
  {"name": "cmr", "type": "diagnostic_test"},
  {"name": "lge", "type": "clinical_finding"},
  {"name": "left ventricle", "type": "anatomical_structure"}
]

Do NOT output:
[
  {"name": "cardiac magnetic resonance", "type": "imaging_modality"},
  {"name": "late gadolinium enhancement", "type": "clinical_finding"},
  {"name": "imaging", "type": "imaging_modality"}
]

Reason:
- CMR and LGE should not be expanded because their full forms are not explicit in this section.
- acronym expansion is handled later by deterministic validation using the document acronym cache.
- imaging is only a broad section heading.


Example 4:
Input text:
\"\"\"
Title: Treatment

Body:
In hypertrophic cardiomyopathy, severe left ventricular outflow tract obstruction may cause syncope.
TAVI may be considered in selected patients.
\"\"\"

Correct output:
[
  {"name": "hypertrophic cardiomyopathy", "type": "disease"},
  {"name": "left ventricular outflow tract obstruction", "type": "clinical_finding"},
  {"name": "syncope", "type": "clinical_finding"},
  {"name": "tavi", "type": "procedure_or_intervention"}
]

Do NOT output:
[
  {"name": "treatment", "type": "care_strategy"},
  {"name": "severe", "type": "clinical_finding"},
  {"name": "left ventricular outflow tract", "type": "anatomical_structure"},
  {"name": "obstruction", "type": "clinical_finding"},
  {"name": "patient", "type": "clinical_finding"},
  {"name": "transcatheter aortic valve implantation", "type": "procedure_or_intervention"}
]

Reason:
- treatment is only a broad section heading.
- severe is only a standalone modifier.
- left ventricular outflow tract obstruction is the full clinical finding; do not split it into anatomical fragments.
- patient is generic.
- TAVI should not be expanded unless the full form is explicit in the text.


Example 5:
Input text:
\"\"\"
Title: Prognosis

Body:
Late gadolinium enhancement is associated with increased risk of cardiovascular death, heart failure hospitalization, and major bleeding.
\"\"\"

Correct output:
[
  {"name": "late gadolinium enhancement", "type": "clinical_finding"},
  {"name": "cardiovascular death", "type": "clinical_outcome"},
  {"name": "heart failure hospitalization", "type": "clinical_outcome"},
  {"name": "major bleeding", "type": "clinical_outcome"}
]

Do NOT output:
[
  {"name": "prognosis", "type": "clinical_outcome"},
  {"name": "risk", "type": "risk_factor"},
  {"name": "outcome", "type": "clinical_outcome"},
  {"name": "event", "type": "clinical_outcome"}
]

Reason:
- cardiovascular death, heart failure hospitalization, and major bleeding are specific clinical outcomes.
- prognosis, risk, outcome, and event are generic words in this context.
"""


ENTITY_EXTRACTION_SINGLE_SYSTEM_PROMPT = f"""You are a medical terminology expert working on cardiology guidelines.

Task:
Extract normalized, reusable cardiology-related concepts from the provided text.

{ENTITY_EXTRACTION_GENERAL_RULES}

{ENTITY_TYPE_GUIDANCE}

{ENTITY_DISAMBIGUATION_GUIDANCE}

{ENTITY_EXTRACTION_EXAMPLES}
"""


ENTITY_EXTRACTION_BATCH_SYSTEM_PROMPT = f"""You are a medical terminology expert working on cardiology guidelines.

Task:
For each provided section, extract normalized, reusable cardiology-related concepts.

{ENTITY_EXTRACTION_GENERAL_RULES}

Batch-specific rules:
- Keep concepts separated by section uid.
- Return every provided uid exactly once, even if its concept list is empty.
- Do NOT copy concepts from one section to another unless they are explicitly supported in that same section.
- Do NOT use information from one section to expand an acronym in another section.
- The text of each section is independent evidence for that section's concepts.

{ENTITY_TYPE_GUIDANCE}

{ENTITY_DISAMBIGUATION_GUIDANCE}

{ENTITY_EXTRACTION_EXAMPLES}

Expected batch format:
[
  {{
    "uid": "section_uid_1",
    "concepts": [
      {{"name": "hypertrophic cardiomyopathy", "type": "disease"}},
      {{"name": "left ventricular outflow tract obstruction", "type": "clinical_finding"}}
    ]
  }},
  {{
    "uid": "section_uid_2",
    "concepts": []
  }}
]
"""


def build_entity_extraction_single_user_prompt(text: str) -> str:
    return f'''
Text:
"""
{text}
"""

Return a valid JSON array only, using this format:
[
  {{"name": "hypertrophic cardiomyopathy", "type": "disease"}},
  {{"name": "left ventricular outflow tract obstruction", "type": "clinical_finding"}},
  {{"name": "troponin", "type": "biomarker"}},
  {{"name": "genetic testing", "type": "diagnostic_test"}},
  {{"name": "cardiac magnetic resonance", "type": "imaging_modality"}},
  {{"name": "tavi", "type": "procedure_or_intervention"}}
]
'''.strip()


def build_entity_extraction_batch_user_prompt(
    sections_payload: List[Dict[str, Any]]
) -> str:
    return (
        "Sections:\n"
        + json.dumps(sections_payload, ensure_ascii=False, indent=2)
        + "\n\nReturn valid JSON only. Use exactly one object per uid and include all uids."
    )