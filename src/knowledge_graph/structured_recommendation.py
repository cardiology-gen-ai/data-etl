from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from knowledge_graph.entity_schema import ALLOWED_TYPES


_ALLOWED_TYPES_SORTED = tuple(sorted(ALLOWED_TYPES))
EntityTypeLiteral = Literal[_ALLOWED_TYPES_SORTED]  # type: ignore[misc,valid-type]


class Qualifier(BaseModel):
    severity: Optional[str] = Field(
        None, description="e.g. 'severe', 'low-grade', 'symptomatic'."
    )
    duration: Optional[str] = Field(
        None, description="e.g. '>10 years', 'longstanding'."
    )
    confirmation: Optional[str] = Field(
        None,
        description=(
            "How the condition must be confirmed, e.g. 'confirmed by a "
            "second pathologist', 'on at least two separate endoscopies'."
        ),
    )
    min_count: Optional[int] = Field(
        None, description="Numerical minimum if expressed (e.g. 2 endoscopies)."
    )
    threshold: Optional[str] = Field(
        None, description="Numerical threshold (e.g. '>=35 kg/m^2', 'HbA1c >=7%')."
    )
    dose: Optional[str] = Field(
        None,
        description=(
            "Verbatim dosage/regimen attached to an action (e.g. '10 mg once daily', 'titrated to maximally tolerated dose'). "
            "Use this when the dose is a *modifier* of an action whose type is drug_or_drug_class; "
            "if the dose itself is the entity, use type=dose_or_regimen instead."
        ),
    )
    route: Optional[str] = Field(
        None, description="e.g. 'oral', 'intravenous', 'subcutaneous'."
    )
    negated: bool = Field(
        False,
        description=(
            "True for inline negation (e.g. 'without visible lesions'). Use Population.excludes for top-level negation."
        ),
    )


class EntityTerm(BaseModel):
    """A clinical entity playing a role in a recommendation.

    Used for both population terms (requires/excludes) and action terms (intervention/alternatives/purpose).
    The verbatim contract applies to ``entity_text``; ``type`` is the local entity-schema label assigned by
    the LLM and must be one of ``kg.entity_schema.ALLOWED_TYPES``.
    """

    entity_text: str = Field(
        description="Verbatim span from the recommendation. Do NOT paraphrase."
    )
    type: EntityTypeLiteral = Field(
        description=(
            "Local entity-schema label for entity_text. Must be one of the "
            "ALLOWED_TYPES values; the prompt lists each with a short "
            "definition. Pick the most specific applicable type."
        )
    )
    qualifiers: Qualifier = Field(default_factory=Qualifier)


# Backward-compatibility alias. Older code may still import ConditionTerm.
ConditionTerm = EntityTerm


class Population(BaseModel):
    requires: list[EntityTerm] = Field(default_factory=list)
    excludes: list[EntityTerm] = Field(default_factory=list)
    logical_operator: Literal["AND", "OR"] = "AND"


class Action(BaseModel):
    intervention: EntityTerm = Field(
        description="Main intervention as a typed EntityTerm."
    )
    alternatives: list[EntityTerm] = Field(
        default_factory=list,
        description="Typed alternative interventions for 'X and/or Y'.",
    )
    purpose: Optional[EntityTerm] = Field(
        None,
        description=(
            "Verbatim purpose clause after 'to ...' / 'in order to ...', "
            "typed as a clinical_outcome when it expresses an aim (mortality, "
            "symptom relief, hospitalisation, ...)."
        ),
    )


class RecommendationStructure(BaseModel):
    modality: Literal[
        "recommend", "should_consider", "may_consider", "recommend_against"
    ]
    population: Population
    action: Action
    extraction_notes: Optional[str] = None

_TYPE_GLOSSARY = """\
- disease: named diseases or syndromes (e.g. "HFrEF", "atrial fibrillation")
- clinical_finding: signs, symptoms, phenotypes (e.g. "dyspnoea on exertion")
- risk_factor: modifiable or fixed risk exposures (e.g. "smoking", "obesity")
- genetic_factor: genes, variants, mutations (e.g. "LDLR variant")
- biomarker: measured analytes / lab values (e.g. "NT-proBNP", "LDL-C")
- diagnostic_test: lab or functional tests as actions (e.g. "ECG", "stress test")
- imaging_modality: imaging acts/devices (e.g. "TTE", "cardiac MRI")
- score_or_risk_model: scores/calculators (e.g. "HAS-BLED", "SCORE2")
- drug_or_drug_class: specific drugs or pharmacological classes (e.g. \
"bisoprolol", "SGLT2 inhibitors")
- procedure_or_intervention: therapeutic procedures (e.g. "CABG", "ablation")
- device: implantable or wearable devices (e.g. "ICD", "CRT-D")
- complication_or_comorbidity: comorbid conditions framed as complications
- care_strategy: management/follow-up strategies (e.g. "lifestyle modification")
- anatomical_structure: body parts or sites (e.g. "left atrial appendage")
- clinical_outcome: aims / endpoints / events (e.g. "all-cause mortality", \
"heart failure hospitalisation", "symptom relief"). Use this for the purpose \
clause of a recommendation.
- dose_or_regimen: the dose/regimen itself when it is the entity (rare); for \
a dose attached to a drug, prefer setting qualifiers.dose instead.\
"""


SYSTEM_PROMPT = f"""You extract the logical structure of clinical practice \
recommendations from European Society of Cardiology (ESC) guidelines.

You will receive ONE recommendation at a time, together with its Class \
(I, IIa, IIb, III) and Level of evidence (A, B, C) parsed from the source \
table. Populate the structured schema.

INVIOLABLE RULES
1. Every entity_text field MUST be a VERBATIM span copied from the source \
recommendation. Do not paraphrase, do not expand acronyms, do not fix typos \
(the source has OCR artefacts like 'effrts', 'ifestyle' -- keep them).
2. Every entity_text MUST be assigned a `type` chosen from the entity-type \
glossary below. Pick the most specific applicable type; if two types could \
fit, prefer the one closer to the entity's clinical role in this sentence \
(e.g. "ECG" as a diagnostic step -> diagnostic_test; "left ventricle" as a \
location -> anatomical_structure).
3. If a piece of information is not explicit in the text, DO NOT infer it. \
Leave the field empty or null.
4. Determine `modality` from the verb in the recommendation:
   - "is recommended" / "it is recommended" / "must" -> recommend
   - "should be considered" -> should_consider
   - "may be considered" -> may_consider
   - "is not recommended" / "should not" / "must not" -> recommend_against
   The Class column is a useful cross-check (I<->recommend, \
IIa<->should_consider, IIb<->may_consider, III<->recommend_against) but the \
VERB IN THE TEXT is the source of truth -- if they disagree, follow the verb \
and explain in extraction_notes.
5. For "X and/or Y" or "X or Y" in the action, put the first as \
action.intervention and the others in action.alternatives. Each carries its \
own type independently.
6. For numerical qualifiers (>=35 kg/m^2, >=2 endoscopies, HbA1c >7%) put \
them in qualifiers.threshold or qualifiers.min_count. For dose modifiers of \
a drug ("10 mg once daily") put them in qualifiers.dose on the intervention.
7. The eligibility population is everything after "in patients with ...", \
"for individuals ...", "in T2DM ...", etc. Decompose it: each clinical \
concept becomes its own EntityTerm. Example:
   "patients with T2DM without symptomatic ASCVD or severe TOD"
   -> requires: [T2DM (disease)]
   -> excludes: [symptomatic ASCVD (disease, qualifiers.severity='symptomatic'),
                 severe TOD (complication_or_comorbidity, qualifiers.severity='severe')]
8. If the recommendation refers to a section ("-Section 5.1.1"), that suffix \
is metadata, not part of the action. Ignore it.
9. The recommendation text may begin with a group-context clause inherited \
from the section header of the source table (e.g. "Patients with HFrEF. <\
recommendation>..." or "In T2DM with established ASCVD. <recommendation>..."). \
If that clause specifies a population, condition, or eligibility scope, treat \
it AS IF it were written inline in the recommendation and add the relevant \
EntityTerms to Population.requires (or Population.excludes if negated). \
If instead the clause is a mere topic/navigational label (e.g. "Diabetes", \
"Lifestyle interventions", "Pharmacological therapy"), ignore it. Verbatim \
spans extracted from the group-context clause are valid: the source for the \
verbatim rule is the WHOLE input text, prefix included.
10. action.purpose is typically a clinical_outcome ("to reduce mortality") \
or a disease being prevented ("to prevent stroke"). Leave it null if absent.

ENTITY TYPE GLOSSARY
{_TYPE_GLOSSARY}
"""

HUMAN_PROMPT = """Class: {class_}
Level: {level}
Recommendation text: \"\"\"{recommendation}\"\"\"

Extract the structured representation now."""


def build_extraction_chain(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout: float = 60.0,
    max_retries: int = 0,
):
    """Return a runnable: dict(class_, level, recommendation) -> RecommendationStructure."""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    structured_llm = llm.with_structured_output(
        RecommendationStructure, method="function_calling"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    return prompt | structured_llm

_CLASS_TO_MODALITY = {
    "I": "recommend",
    "IIa": "should_consider",
    "IIb": "may_consider",
    "III": "recommend_against",
}


def iter_entity_terms(extraction: RecommendationStructure):
    """Yield (role, index, EntityTerm) for every EntityTerm in the extraction.

    Roles: 'requires', 'excludes', 'intervention', 'alternative', 'purpose'.
    """
    for i, term in enumerate(extraction.population.requires):
        yield "requires", i, term
    for i, term in enumerate(extraction.population.excludes):
        yield "excludes", i, term
    yield "intervention", 0, extraction.action.intervention
    for i, term in enumerate(extraction.action.alternatives):
        yield "alternative", i, term
    if extraction.action.purpose is not None:
        yield "purpose", 0, extraction.action.purpose


def validate_extraction(
    extraction: RecommendationStructure,
    catalog_class: Optional[str],
    source_text: str,
) -> dict[str, str]:
    """Return a dict of flags. Empty dict == clean.

    Flags
    -----
    - modality_mismatch: LLM modality disagrees with Class.
    - non_verbatim_<role>_<i>: extracted span is not a substring of source.
    - bad_type_<role>_<i>: extracted type is not in ALLOWED_TYPES (should
      not happen if the LLM respected the Literal, but kept defensively).
    - empty_population: no requires AND no excludes (often legitimate).
    """
    flags: dict[str, str] = {}

    if catalog_class:
        expected = _CLASS_TO_MODALITY.get(catalog_class)
        if expected and expected != extraction.modality:
            flags["modality_mismatch"] = (
                f"catalog class {catalog_class} implies {expected}, "
                f"LLM extracted {extraction.modality}"
            )

    src = source_text.lower()

    for role, idx, term in iter_entity_terms(extraction):
        span = term.entity_text or ""
        if span and span.lower() not in src:
            flags[f"non_verbatim_{role}_{idx}"] = span
        if term.type not in ALLOWED_TYPES:
            flags[f"bad_type_{role}_{idx}"] = term.type

    if not extraction.population.requires and not extraction.population.excludes:
        flags["empty_population"] = "no eligibility conditions extracted"

    return flags