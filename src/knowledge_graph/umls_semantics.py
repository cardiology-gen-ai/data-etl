"""UMLS semantic type / semantic group helpers.

Two static mappings are bundled here, both small and stable:

1. ``TUI_TO_SEMGROUP``: every UMLS Semantic Type (TUI, ~127 of them) mapped
   to its Semantic Group (one of 15 macro-categories). Source: NLM Semantic
   Network ``SemGroups-v04.txt``.

2. ``ROLE_TO_ACCEPTED_GROUPS``: the *default* contract between the role of
   a span in a ESC recommendation (``requires``, ``intervention``, ...) and
   the Semantic Groups that are plausible for that role. Used to disambiguate
   top-k UMLS candidates returned by scispacy: a span tagged as
   ``intervention`` should prefer a CHEM/PROC/DEVI CUI even if a DISO CUI
   scores higher.
"""

from typing import Iterable, Optional

TUI_TO_SEMGROUP: dict[str, str] = {
    # ACTI -- Activities & Behaviors
    "T052": "ACTI", "T053": "ACTI", "T056": "ACTI", "T051": "ACTI",
    "T064": "ACTI", "T055": "ACTI", "T066": "ACTI", "T057": "ACTI",
    "T054": "ACTI",
    # ANAT -- Anatomy
    "T017": "ANAT", "T029": "ANAT", "T023": "ANAT", "T030": "ANAT",
    "T031": "ANAT", "T022": "ANAT", "T025": "ANAT", "T026": "ANAT",
    "T018": "ANAT", "T021": "ANAT", "T024": "ANAT",
    # CHEM -- Chemicals & Drugs
    "T116": "CHEM", "T195": "CHEM", "T123": "CHEM", "T122": "CHEM",
    "T103": "CHEM", "T120": "CHEM", "T104": "CHEM", "T200": "CHEM",
    "T196": "CHEM", "T126": "CHEM", "T131": "CHEM", "T125": "CHEM",
    "T129": "CHEM", "T130": "CHEM", "T197": "CHEM", "T114": "CHEM",
    "T109": "CHEM", "T121": "CHEM", "T192": "CHEM", "T127": "CHEM",
    # CONC -- Concepts & Ideas
    "T185": "CONC", "T077": "CONC", "T169": "CONC", "T102": "CONC",
    "T078": "CONC", "T170": "CONC", "T171": "CONC", "T080": "CONC",
    "T081": "CONC", "T089": "CONC", "T082": "CONC",
    # DEVI -- Devices
    "T203": "DEVI", "T074": "DEVI", "T075": "DEVI",
    # DISO -- Disorders
    "T020": "DISO", "T190": "DISO", "T049": "DISO", "T019": "DISO",
    "T047": "DISO", "T050": "DISO", "T033": "DISO", "T037": "DISO",
    "T048": "DISO", "T191": "DISO", "T046": "DISO", "T184": "DISO",
    # GENE -- Genes & Molecular Sequences
    "T087": "GENE", "T088": "GENE", "T028": "GENE", "T085": "GENE",
    "T086": "GENE",
    # GEOG -- Geographic Areas
    "T083": "GEOG",
    # LIVB -- Living Beings
    "T100": "LIVB", "T011": "LIVB", "T008": "LIVB", "T194": "LIVB",
    "T007": "LIVB", "T012": "LIVB", "T204": "LIVB", "T099": "LIVB",
    "T013": "LIVB", "T004": "LIVB", "T096": "LIVB", "T016": "LIVB",
    "T015": "LIVB", "T001": "LIVB", "T101": "LIVB", "T002": "LIVB",
    "T098": "LIVB", "T097": "LIVB", "T014": "LIVB", "T010": "LIVB",
    "T005": "LIVB",
    # OBJC -- Objects
    "T071": "OBJC", "T168": "OBJC", "T073": "OBJC", "T072": "OBJC",
    "T167": "OBJC",
    # OCCU -- Occupations
    "T091": "OCCU", "T090": "OCCU",
    # ORGA -- Organizations
    "T093": "ORGA", "T092": "ORGA", "T094": "ORGA", "T095": "ORGA",
    # PHEN -- Phenomena
    "T038": "PHEN", "T069": "PHEN", "T068": "PHEN", "T034": "PHEN",
    "T070": "PHEN", "T067": "PHEN",
    # PHYS -- Physiology
    "T043": "PHYS", "T201": "PHYS", "T045": "PHYS", "T041": "PHYS",
    "T044": "PHYS", "T032": "PHYS", "T040": "PHYS", "T042": "PHYS",
    "T039": "PHYS",
    # PROC -- Procedures
    "T060": "PROC", "T065": "PROC", "T058": "PROC", "T059": "PROC",
    "T063": "PROC", "T062": "PROC", "T061": "PROC",
}


def tuis_to_groups(tuis: Iterable[str]) -> list[str]:
    """Resolve a sequence of TUIs to their (deduplicated, stable-order)
    Semantic Groups. Unknown TUIs are skipped silently."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tuis:
        sg = TUI_TO_SEMGROUP.get(t)
        if sg and sg not in seen:
            seen.add(sg)
            out.append(sg)
    return out


def primary_group(tuis: Iterable[str]) -> Optional[str]:
    """First Semantic Group resolvable from ``tuis``, or None."""
    groups = tuis_to_groups(tuis)
    return groups[0] if groups else None

# Defaults tuned for ESC clinical practice recommendations.
ROLE_TO_ACCEPTED_GROUPS: dict[str, frozenset[str]] = {
    # Eligibility/exclusion conditions: diseases, anatomical sites,
    # physiological findings, labs, prior procedures/devices.
    "requires":     frozenset({"DISO", "ANAT", "PHYS", "CHEM", "PROC", "DEVI"}),
    "excludes":     frozenset({"DISO", "ANAT", "PHYS", "CHEM", "PROC", "DEVI"}),
    # Interventions: drugs, procedures, implantable devices.
    "intervention": frozenset({"CHEM", "PROC", "DEVI"}),
    "alternative":  frozenset({"CHEM", "PROC", "DEVI"}),
    # Outcomes / purposes: clinical events, physiological improvements.
    "purpose":      frozenset({"DISO", "PHYS", "PHEN"}),
}


def is_role_compatible(
    span_role: str,
    semantic_groups: Iterable[str],
    accepted: Optional[dict[str, frozenset[str]]] = None,
) -> bool:
    accepted = accepted or ROLE_TO_ACCEPTED_GROUPS
    allowed = accepted.get(span_role)
    if not allowed:
        return True
    return any(g in allowed for g in semantic_groups)


def expected_semgroups_for_role(role: str) -> list[str]:
    """Return the prior-acceptable SemGroups for an edge role, as a sorted list (Neo4j stores list[str] cleanly).
     Unknown roles get an empty list  meaning "no constraint"."""
    groups = ROLE_TO_ACCEPTED_GROUPS.get(role)
    return sorted(groups) if groups else []


def aggregate_semgroup_priors_for_concept(
        tx, concept_name: str
) -> dict[str, int]:
    """Count how many incoming role edges expect each SemGroup, across all Recommendations that reference this concept.
    """
    result = tx.run(
        """
        MATCH (r:Recommendation)-[e]->(c:Concept {name: $name})
        WHERE e.expected_semgroups IS NOT NULL
        UNWIND e.expected_semgroups AS sg
        RETURN sg AS semgroup, count(*) AS support
        ORDER BY support DESC, semgroup ASC
        """,
        name=concept_name,
    )
    return {record["semgroup"]: int(record["support"]) for record in result}


def score_candidate_with_role_prior(
        candidate_semgroups: Iterable[str],
        role_priors: dict[str, int],
        base_score: float,
        bonus: float = 0.05,
) -> float:
    """Adjust a UMLS candidate score by a small bonus per SemGroup match.

    Designed to be plugged into ``umls_normalization.select_best_*`` as a second-pass tiebreaker.

    Defaults to a small additive bonus (+0.05 per supported SemGroup, weighted by support count),
    small enough not to override a clearly better lexical/embedding match but enough to break near-ties.
    """
    if not role_priors:
        return base_score
    total_support = sum(role_priors.values()) or 1
    bonus_sum = 0.0
    for sg in candidate_semgroups:
        if sg in role_priors:
            bonus_sum += bonus * (role_priors[sg] / total_support)
    return base_score + bonus_sum