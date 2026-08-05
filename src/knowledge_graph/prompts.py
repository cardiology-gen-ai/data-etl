"""
prompts.py

Prompt definitions for extracting clinical entities from cardiology guideline
sections.

The allowed entity types and their definitions are imported from
entity_schema.py so the structured-output schema, deterministic validation,
and prompt guidance share one source of truth.

This extractor identifies concepts and their intrinsic semantic types only.
Contextual roles such as risk factor, comorbidity, complication, outcome, or
target_population, and qualifiers such as negation, severity, thresholds,
duration, or frequency, are intentionally deferred to separate enrichment
stages. A population_or_patient_group is an entity type; target_population is a
future contextual role that such an entity may play in a recommendation or
clinical assertion.

Acronym policy:
- the LLM must not guess expansions that are absent from the current section;
- acronym-only outputs preserve the exact source capitalization;
- deterministic validation may subsequently expand a short form through the
  document acronym cache, but only when the short form occurs in the section
  and the cached mapping passes the acronym validation rules.
"""

import json
from typing import Any, Dict, List

from knowledge_graph.entity_schema import TYPE_DEFINITIONS


def _build_entity_type_guidance() -> str:
    """Build prompt guidance directly from the canonical local entity schema."""
    lines = ["Allowed entity types:"]
    lines.extend(
        f"- {entity_type}: {definition}"
        for entity_type, definition in TYPE_DEFINITIONS.items()
    )
    return "\n".join(lines)


ENTITY_TYPE_GUIDANCE = _build_entity_type_guidance()


ENTITY_DISAMBIGUATION_GUIDANCE = """Disambiguation guidance:
- Assign the intrinsic type of the concept, not the role that it plays in the local sentence.
- Diseases remain disease even when described as risk factors, comorbidities, complications, indications, contraindications, or outcomes.
- Use exposure_or_lifestyle_factor only for behavioural, lifestyle, environmental, occupational, substance-use, or pollutant exposures. A disease or clinical finding does not change to this type merely because it increases risk.
- When the complete phrase describes use, abuse, smoking, vaping, consumption, or exposure, prefer exposure_or_lifestyle_factor. The named substance by itself may still be drug_or_drug_class when the substance itself is the reusable concept.
- Environmental constituents, emissions, particulate matter, and inhaled pollutants are exposures, not biomarkers. A biomarker must be a biological analyte or marker measured in a patient or biological sample.
- Use population_or_patient_group for an explicitly named demographic, life-stage, familial, reproductive, occupational, survivorship, genotype-defined, or otherwise clinically meaningful group of people. Examples include children, adolescents, older adults, neonates, fetuses or foetuses, pregnant women, athletes, first-degree relatives, and childhood cancer survivors. Do not extract generic patient references or a disease name or acronym followed only by patient or patients when the disease itself is the reusable concept.
- Care settings and service locations such as primary care, secondary care, intensive care, outpatient clinics, hospitals, wards, practices, and care units are not population_or_patient_group. Omit them because the current schema has no care-setting entity type.
- Scientific societies, councils, associations, colleges, committees, task forces, working groups, professional organizations, and care teams are not patient populations and have no entity type in the current schema. Omit names such as European Society of Cardiology, Heart Failure Association, or ESC Council of Cardio-Oncology.
- Medical disciplines and specialties such as cardiology, oncology, haematology, and cardio-oncology are not diseases, populations, or care strategies. Omit the discipline itself; a specific service, programme, test, disease, or treatment may still be extracted.
- Distinguish an entity type from a future contextual role: a group may have intrinsic type population_or_patient_group and later receive the role target_population in a separate recommendation or assertion extraction stage.
- A physiological state is not automatically a population. Pregnancy may be clinical_finding when the state itself is the concept, whereas pregnant women is population_or_patient_group.
- Use diagnostic_test for a named diagnostic or monitoring examination, including imaging, laboratory, functional, electrophysiological, and genetic tests.
- Research-only methods, study designs, evidence-source labels, database concepts, and population-level statistics are not clinical diagnostic tests or patient findings. Omit randomized controlled trials, observational studies, registries, meta-analyses, genome-wide association studies, allele-frequency statistics, and generic variables such as tumour type or sex category.
- Distinguish an examination from its result. The examination or acquisition method is diagnostic_test; an observation, pattern, enhancement, defect, abnormality, measurement, value, or result produced by it is clinical_finding.
- Distinguish a test from a care process. A named examination or testing method is diagnostic_test; an organized programme or longitudinal strategy that determines whom to test, when to repeat testing, or how to extend testing through a family or population is care_strategy.
- Use procedure_or_intervention for a performed therapeutic or supportive intervention, including chemotherapy or radiotherapy as treatment modalities, anaesthesia, sedation, surgery, catheter procedures, ablation, and other clinical procedures. Do not classify an intervention as clinical_finding.
- A treatment modality is not exposure_or_lifestyle_factor. Do not classify chemotherapy, radiotherapy, anticancer therapy, or cardiotoxic therapy as a lifestyle or environmental exposure merely because the treatment exposes the patient to toxicity.
- Use clinical_finding for patient-level symptoms, signs, phenotypes, physiological or pathological states, measured results, and ECG/imaging/laboratory findings. Technical artefacts, equipment limitations, acquisition problems, and image-quality defects are not patient findings and should normally not be extracted as clinical entities.
- Use disease only for a named disease, syndrome, or disorder. Descriptive damage, dysfunction, impairment, abnormality, defect, reduced function, or altered function is generally clinical_finding unless the complete phrase is an established named disease or syndrome in the section.
- Use biomarker only for a specific biological analyte, molecule, cellular marker, or measurable biological substance assessed in a biological specimen. Generic category phrases such as cardiac biomarker, serum biomarker, circulating marker, or cardiac biomarkers are not sufficiently specific and should be omitted. Imaging-derived measurements, physiological measurements, functional indices, pressures, fractions, scores, environmental pollutants, and exposure constituents are not biomarkers. Use diagnostic_test for the examination that measures a biomarker, and clinical_finding for an explicitly stated patient-level result or value.
- Use anatomical_structure only for body structures, organs, chambers, vessels, valves, tissues, and anatomical sites. People, patient groups, life stages, fetuses, and neonates are not anatomical structures.
- Use clinical_outcome only for endpoints describing what happens to a patient or population, such as mortality, hospitalization, recurrence, quality of life, functional improvement, or symptom improvement. Decisions, diagnostic labels, management activities, treatment selection, counselling, testing, screening, and surveillance are not clinical outcomes. A disease does not change type merely because it is evaluated as an outcome.
- Use care_strategy only for a specific reusable programme, pathway, or coordinated care process such as family screening, cascade testing, genetic counselling, structured follow-up, rehabilitation, surveillance, patient education, shared decision-making, or a strategy that determines whom to test, how testing is propagated through a family, when it is repeated, or how results guide longitudinal care. A single examination remains diagnostic_test. Do not use care_strategy for generic management, treatment, therapy, care, monitoring, screening, or follow-up alone.
- Use microorganism_or_pathogen for a named microorganism or a clinically meaningful microorganism group. Do not extract generic words such as bacteria, virus, pathogen, or microorganism by themselves.
- Use genetic_factor for a specific gene, mutation, variant, genotype, inherited molecular factor, or explicit genetic status. A merely adjectival description of causation, inheritance, or molecular origin is not a genetic entity.
- Canonical gene symbols such as SCN5A, MYBPC3, TTN, TTR, or TMEM43 are genetic_factor entities, not clinical acronyms. Return the exact uppercase symbol from the source and do not replace it with a guessed gene or protein long form.
- Never classify a generic causal or aetiological description as disease. Disease is reserved for a named disease, syndrome, or disorder. Phrases whose semantic head is only cause, aetiology, etiology, mechanism, origin, or basis normally describe an explanation rather than a reusable clinical entity and should be omitted unless they contain a separately named entity that can be extracted on its own.
- Use device only for a clinical or medical device. Consumer products, exposure sources, and substance-delivery products are not medical devices merely because they are physical objects.
- Use drug_or_drug_class for a named drug, therapeutic agent, or clinically meaningful drug class. Do not extract generic phrases such as medication, cardiovascular medication, cancer therapy, anticancer treatment, pharmacological therapy, immunosuppression, or drug treatment. A treatment modality such as radiotherapy is procedure_or_intervention, not a drug.
- Use procedure_or_intervention for a performed therapeutic procedure or intervention, and device for the physical medical device itself.
"""


ENTITY_EXTRACTION_GENERAL_RULES = """General rules:
- Extract normalized, reusable clinical concepts explicitly supported by the provided section.
- Keep extracted wording close to an explicit source span. Do not append a missing generic head such as therapy, treatment, disease, syndrome, matter, patients, or population, and do not split a coordinated phrase into normalized concepts that are not individually stated.
- Use lowercase concept names, except acronym-only outputs and canonical gene symbols, which must preserve the exact capitalization used in the source.
- Use singular form when appropriate.
- Prefer standard clinical terminology while preserving the meaning of the explicit source span.
- Do NOT invent, infer, or add concepts that are only implied.
- Do NOT return duplicate concepts.
- Each extracted concept must have exactly one canonical type from the allowed types.
- Prefer the most specific clinically meaningful span explicitly present in the text.
- Do NOT split one meaningful concept into smaller fragments when the longer expression is the true concept.
- Do NOT add generic parent concepts unless they are separately and explicitly meaningful in the text.
- Do NOT extract generic or trivial words such as management, treatment, therapy, recommendation, patient, population, disease, condition, risk, test, procedure, intervention, medication, care, finding, outcome, event, score, model, pathogen, or microorganism.
- Do NOT extract scientific organizations, professional bodies, medical disciplines, research designs, evidence-source labels, generic categories, or isolated adjectives as clinical entities.
- Do NOT extract broad organizational headings such as diagnosis, management, follow-up, screening, prognosis, risk stratification, imaging, genetics, prevention, pregnancy, or therapy unless the title itself names a specific reusable clinical concept.
- Do NOT extract abstract actions or care verbs such as evaluate, consider, recommend, screen, monitor, assess, manage, or treat unless they are part of a specific named concept.
- Do NOT extract broad process phrases headed by diagnosis, management, selection, treatment, medication, testing, screening, surveillance, monitoring, counselling, assessment, or evaluation unless the complete phrase identifies a specific reusable examination, programme, pathway, or care strategy.
- In particular, omit generic "clinical evaluation", "clinical assessment", "general evaluation", and "routine assessment". Keep specific examinations such as physical examination, electrocardiography, echocardiography, or genetic testing.
- Do NOT extract recommendation or evidence labels such as Class I, Class IIa, Class IIb, Class III, Level A, Level B, or Level C.
- Do NOT extract standalone modifiers such as mild, moderate, severe, symptomatic, asymptomatic, primary, secondary, advanced, acute, chronic, recurrent, persistent, isolated, or familial.
- Omit contextual severity or status modifiers from the canonical concept name when the base concept remains clinically meaningful. Retain a modifier only when it is part of an established named disease, subtype, population, or other reusable concept.
- Extract a population or patient group only when the complete phrase adds an independent demographic, life-stage, familial, reproductive, occupational, survivorship, genotype, or other clinically meaningful distinction. Do not extract a disease name or acronym followed only by patient or patients when the disease itself is already the reusable concept.
- Do NOT extract contextual roles as concepts, including risk factor, comorbidity, complication, indication, contraindication, outcome role, or target population role.
- Do NOT extract standalone contextual qualifiers such as negation, severity, thresholds, duration, frequency, age constraints, sex constraints, pregnancy context, or temporal context. A complete population phrase may still be extracted as population_or_patient_group.
- Do NOT extract generic causal or aetiological descriptions whose main meaning is only cause, aetiology, etiology, mechanism, origin, or basis. Extract an explicitly named disease, gene, variant, exposure, or other clinical entity contained in the phrase instead; otherwise omit the phrase.
- Do NOT extract unspecified findings, unspecified abnormalities, unspecified variants, or generic technical artefacts as standalone concepts.
- Merge synonyms only when their equivalence is explicit in the same section.
- Do NOT guess an abbreviation or acronym expansion that is absent from the current section.
- If only a clinically meaningful acronym appears, return the exact source short form with its original capitalization, such as "CMR", rather than a lowercased form or a guessed expansion. Deterministic validation may later expand it through the document acronym cache when safe.
- Preserve an undefined acronym when it is embedded inside a larger explicit phrase. Return "12-lead ECG", "HAS-BLED score", "CV surveillance", "AF ablation", or "intensity-modulated RT" exactly as written rather than inventing a long-form phrase absent from the section.
- Preserve classifier-like acronym phrases such as "ICI myocarditis", "RAF inhibitor", and "MEK inhibitor". Do not mechanically rewrite them as unnatural concatenations such as "immune checkpoint inhibitors myocarditis" or long pathway-name-plus-inhibitor phrases.
- For a named risk score whose source uses only the acronym, return the conventional form "<ACRONYM> score", for example "HAS-BLED score".
- If both an acronym and its full form are explicitly provided in the same section, prefer the full form and do not return the acronym as a duplicate concept.
- Be conservative with short forms that can also be ordinary words, such as AS or MR. Return them only when the source clearly uses the uppercase form as a medical abbreviation.
- Return valid JSON only.
- Use double quotes in JSON.
- Do not add explanations, markdown fences, or any text outside the requested JSON object.
"""


ENTITY_EXTRACTION_EXAMPLES = """Examples:

Example 1 — generic title only
Input text:
\"\"\"
Title: Diagnosis
\"\"\"

Concepts:
[]

Do not extract diagnosis: it is only a broad organizational heading.


Example 2 — disease, finding, test, biomarker, and explicit acronym definition
Input text:
\"\"\"
Title: Definitions

Body:
Acute coronary syndrome (ACS) may present with chest pain, changes on a 12-lead electrocardiogram, and elevated cardiac troponin. A final diagnosis may be acute myocardial infarction or unstable angina.
\"\"\"

Concepts:
[
  {"name": "acute coronary syndrome", "type": "disease"},
  {"name": "chest pain", "type": "clinical_finding"},
  {"name": "12-lead electrocardiogram", "type": "diagnostic_test"},
  {"name": "cardiac troponin", "type": "biomarker"},
  {"name": "acute myocardial infarction", "type": "disease"},
  {"name": "unstable angina", "type": "disease"}
]

Do not additionally extract ACS because the full form is explicit. Do not extract changes, elevated, diagnosis, or patient as standalone concepts.


Example 3 — unexpanded acronyms and imaging collapsed into diagnostic_test
Input text:
\"\"\"
Title: Imaging

Body:
CMR showed LGE in the left ventricle. No full acronym definitions are provided in this section.
\"\"\"

Concepts:
[
  {"name": "CMR", "type": "diagnostic_test"},
  {"name": "LGE", "type": "clinical_finding"},
  {"name": "left ventricle", "type": "anatomical_structure"}
]

Preserve the acronym capitalization because no expansion is explicit in this section. Downstream deterministic validation may expand a short form through the document acronym cache when the short form is present and the mapping is safe. Do not extract imaging as a concept.


Example 4 — intrinsic type versus contextual role
Input text:
\"\"\"
Title: Cardiovascular risk

Body:
Hypertension, diabetes mellitus, chronic kidney disease, smoking, and physical inactivity increase cardiovascular risk. Stroke is an important outcome to prevent.
\"\"\"

Concepts:
[
  {"name": "hypertension", "type": "disease"},
  {"name": "diabetes mellitus", "type": "disease"},
  {"name": "chronic kidney disease", "type": "disease"},
  {"name": "smoking", "type": "exposure_or_lifestyle_factor"},
  {"name": "physical inactivity", "type": "exposure_or_lifestyle_factor"},
  {"name": "stroke", "type": "disease"}
]

Do not classify diseases as exposure_or_lifestyle_factor or clinical_outcome merely because the sentence gives them a risk-factor or outcome role. Do not extract risk or outcome by themselves.


Example 5 — microorganism and treatment concepts
Input text:
\"\"\"
Title: Infective endocarditis

Body:
Infective endocarditis caused by Staphylococcus aureus may require intravenous flucloxacillin and valve surgery.
\"\"\"

Concepts:
[
  {"name": "infective endocarditis", "type": "disease"},
  {"name": "staphylococcus aureus", "type": "microorganism_or_pathogen"},
  {"name": "flucloxacillin", "type": "drug_or_drug_class"},
  {"name": "valve surgery", "type": "procedure_or_intervention"}
]

Do not extract caused by, require, treatment, pathogen, or bacteria as concepts.


Example 6 — full clinical finding and procedure acronym
Input text:
\"\"\"
Title: Treatment

Body:
In hypertrophic cardiomyopathy, severe left ventricular outflow tract obstruction may cause syncope. TAVI may be considered in selected patients.
\"\"\"

Concepts:
[
  {"name": "hypertrophic cardiomyopathy", "type": "disease"},
  {"name": "left ventricular outflow tract obstruction", "type": "clinical_finding"},
  {"name": "syncope", "type": "clinical_finding"},
  {"name": "TAVI", "type": "procedure_or_intervention"}
]

Do not include severe in the canonical concept name because it is a contextual severity qualifier. Do not split the obstruction into anatomical fragments. Preserve TAVI exactly because its full form is not explicit in the section. Do not extract selected patients.


Example 7 — care strategy and outcome-like endpoints
Input text:
\"\"\"
Title: Follow-up

Body:
Family screening, genetic counselling, and structured follow-up should be offered. The programme aims to reduce cardiovascular mortality and heart failure hospitalization and to improve quality of life.
\"\"\"

Concepts:
[
  {"name": "family screening", "type": "care_strategy"},
  {"name": "genetic counselling", "type": "care_strategy"},
  {"name": "structured follow-up", "type": "care_strategy"},
  {"name": "cardiovascular mortality", "type": "clinical_outcome"},
  {"name": "heart failure hospitalization", "type": "clinical_outcome"},
  {"name": "quality of life", "type": "clinical_outcome"}
]

Do not extract follow-up, programme, outcome, or improvement as generic standalone concepts.


Example 8 — population, test, care strategy, and procedure
Input text:
\"\"\"
Title: Assessment across life stages

Body:
Neonates and children with an inherited cardiac disorder may undergo genetic testing. First-degree relatives may enter a cascade screening programme. Pregnant women requiring a procedure may receive general anaesthesia.
\"\"\"

Concepts:
[
  {"name": "neonate", "type": "population_or_patient_group"},
  {"name": "children", "type": "population_or_patient_group"},
  {"name": "inherited cardiac disorder", "type": "disease"},
  {"name": "genetic testing", "type": "diagnostic_test"},
  {"name": "first-degree relative", "type": "population_or_patient_group"},
  {"name": "cascade screening programme", "type": "care_strategy"},
  {"name": "pregnant women", "type": "population_or_patient_group"},
  {"name": "general anaesthesia", "type": "procedure_or_intervention"}
]

Do not classify people or life stages as anatomical structures. Do not extract procedure, programme, or patients as generic concepts.


Example 9 — examination results, exposures, biomarkers, and medical devices
Input text:
\"\"\"
Title: Imaging and exposure assessment

Body:
Cardiac magnetic resonance demonstrated late gadolinium enhancement. Blood cardiac troponin was measured. Tobacco smoking and fine particulate matter exposure were recorded. An implantable cardioverter-defibrillator was present.
\"\"\"

Concepts:
[
  {"name": "cardiac magnetic resonance", "type": "diagnostic_test"},
  {"name": "late gadolinium enhancement", "type": "clinical_finding"},
  {"name": "cardiac troponin", "type": "biomarker"},
  {"name": "tobacco smoking", "type": "exposure_or_lifestyle_factor"},
  {"name": "fine particulate matter exposure", "type": "exposure_or_lifestyle_factor"},
  {"name": "implantable cardioverter-defibrillator", "type": "device"}
]

The imaging examination is a diagnostic_test, whereas its observed enhancement is a clinical_finding. A biological analyte measured in a specimen may be a biomarker; a functional fraction, physiological measurement, environmental pollutant, or exposure constituent is not. Device is reserved for the medical device.


Example 10 — organizations, disciplines, treatment modalities, and embedded acronyms
Input text:
\"\"\"
Title: Surveillance and treatment

Body:
The European Society of Cardiology recommends CV surveillance with a 12-lead ECG. HAS-BLED score may support bleeding-risk assessment. Cardio-oncology teams coordinate care during intensity-modulated RT and anthracycline chemotherapy.
\"\"\"

Concepts:
[
  {"name": "CV surveillance", "type": "care_strategy"},
  {"name": "12-lead ECG", "type": "diagnostic_test"},
  {"name": "HAS-BLED score", "type": "score_or_risk_model"},
  {"name": "intensity-modulated RT", "type": "procedure_or_intervention"},
  {"name": "anthracycline chemotherapy", "type": "procedure_or_intervention"}
]

Do not extract European Society of Cardiology, cardio-oncology, teams, discipline, or treatment as standalone concepts. Preserve embedded acronyms exactly when their long forms are not explicitly written in the section.
"""


SINGLE_RESPONSE_FORMAT = """Required response format:
{
  "concepts": [
    {"name": "hypertrophic cardiomyopathy", "type": "disease"},
    {"name": "cardiac magnetic resonance", "type": "diagnostic_test"}
  ]
}

When no valid concept is present, return:
{
  "concepts": []
}
"""


BATCH_RESPONSE_FORMAT = """Required response format:
{
  "sections": [
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
}
"""


ENTITY_EXTRACTION_SINGLE_SYSTEM_PROMPT = f"""You are a medical terminology expert working on cardiology clinical practice guidelines.

Task:
Extract normalized, reusable clinical concepts from the provided section and assign each concept its intrinsic semantic type.

This task extracts entities only. Do not extract recommendation structure, contextual roles, relations, assertions, or qualifiers.

{ENTITY_EXTRACTION_GENERAL_RULES}

{ENTITY_TYPE_GUIDANCE}

{ENTITY_DISAMBIGUATION_GUIDANCE}

{ENTITY_EXTRACTION_EXAMPLES}

{SINGLE_RESPONSE_FORMAT}
"""


ENTITY_EXTRACTION_BATCH_SYSTEM_PROMPT = f"""You are a medical terminology expert working on cardiology clinical practice guidelines.

Task:
For each provided section, extract normalized, reusable clinical concepts and assign each concept its intrinsic semantic type.

This task extracts entities only. Do not extract recommendation structure, contextual roles, relations, assertions, or qualifiers.

{ENTITY_EXTRACTION_GENERAL_RULES}

Batch-specific rules:
- Keep concepts separated by section uid.
- Return every provided uid exactly once, even when its concept list is empty.
- Do not return any uid that was not provided.
- Do not copy concepts from one section to another unless they are explicitly supported in that same section.
- Do not use information from one section to expand an acronym in another section.
- Treat the text of each section as independent evidence for that section's concepts.

{ENTITY_TYPE_GUIDANCE}

{ENTITY_DISAMBIGUATION_GUIDANCE}

{ENTITY_EXTRACTION_EXAMPLES}

{BATCH_RESPONSE_FORMAT}
"""


def build_entity_extraction_single_user_prompt(text: str) -> str:
    """Build the user message for one section."""
    return f'''Section text:
"""
{text}
"""

Return one valid JSON object using exactly this top-level structure:
{{
  "concepts": [
    {{"name": "concept name", "type": "allowed_entity_type"}}
  ]
}}

Return {{"concepts": []}} when the section contains no valid concept.
'''.strip()


def build_entity_extraction_batch_user_prompt(
    sections_payload: List[Dict[str, Any]],
) -> str:
    """Build the user message for a batch of independently evaluated sections."""
    return (
        "Sections:\n"
        + json.dumps(sections_payload, ensure_ascii=False, indent=2)
        + "\n\nReturn one valid JSON object with the top-level key \"sections\". "
        "Include every provided uid exactly once and do not include any other uid."
    )
