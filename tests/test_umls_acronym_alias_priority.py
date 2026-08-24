from knowledge_graph.umls_normalization import (
    SEMANTIC_COMPATIBLE,
    SEMANTIC_INCOMPATIBLE,
    UMLSMatch,
    select_best_umls_api_match,
    select_best_umls_api_match_with_trace,
)


class FakeClient:
    def __init__(self, by_alias):
        self.by_alias = by_alias

    def search_alias(self, alias, search_type, canonical_type=None):
        if search_type != 'exact':
            return None
        return self.by_alias.get(alias)

    def search_alias_candidates(self, alias, search_type, canonical_type=None, trace_limit=3):
        match = self.search_alias(alias, search_type, canonical_type=canonical_type)
        if match is None:
            return None, []
        row = {
            'query_alias': alias,
            'search_type': search_type,
            'api_rank': match.api_rank,
            'cui': match.cui,
            'canonical_name': match.canonical_name,
            'semantic_types': match.semantic_types,
            'lexical_score': match.score,
            'adjusted_score': match.score,
            'selection_score': match.selection_score or match.score,
            'semantic_compatibility': match.semantic_compatibility,
            'type_compatible': match.type_compatible,
            'selection_eligible': match.type_compatible is not False,
            'exclusion_reason': None if match.type_compatible is not False else 'strong_semantic_incompatibility',
            'matched_atom_name': match.matched_atom_name,
            'matched_atom_source': match.matched_atom_source,
            'matched_atom_term_type': match.matched_atom_term_type,
            'matched_atom_score': match.matched_atom_score,
            'matched_atom_count': match.matched_atom_count,
            'matched_atom_source_count': match.matched_atom_source_count,
            'synonym_supported': match.synonym_supported,
            'selected_for_search_strategy': True,
            'selected_for_alias': False,
            'selected_final': False,
            'retained_reason': 'test',
        }
        return match, [row]


def make_match(*, alias, cui, name, score, compatible=True, synonym_supported=True):
    return UMLSMatch(
        alias=alias,
        cui=cui,
        canonical_name=name,
        definition=None,
        aliases=[],
        score=score,
        semantic_types=['Gene or Genome'] if compatible else ['Disease or Syndrome'],
        search_type='exact',
        type_compatible=True if compatible else False,
        semantic_compatibility=SEMANTIC_COMPATIBLE if compatible else SEMANTIC_INCOMPATIBLE,
        api_rank=1,
        selection_score=score,
        canonical_type='genetic_factor',
        matched_atom_name=alias.upper(),
        matched_atom_source='HGNC',
        matched_atom_term_type='ACR',
        matched_atom_score=1.0 if synonym_supported else None,
        matched_atom_count=2 if synonym_supported else 0,
        matched_atom_source_count=2 if synonym_supported else 0,
        synonym_supported=synonym_supported,
    )


def primary_provenance(alias, short):
    return {
        'alias': alias,
        'alias_index': 0,
        'alias_sources': ['document_acronym_expansion', 'document_acronym_expansion_canonicalized'],
        'alias_doc_ids': ['Cardiomyopathies_2023'],
        'acronym_shorts': [short],
    }


def concept_name_provenance(alias, index=1):
    return {
        'alias': alias,
        'alias_index': index,
        'alias_sources': ['concept_name'],
        'alias_doc_ids': [],
        'acronym_shorts': [],
    }


def test_ftx_primary_long_form_beats_conflicting_bare_symbol():
    aliases = ['frataxin', 'ftx']
    provenance = [primary_provenance('frataxin', 'FTX'), concept_name_provenance('ftx')]
    client = FakeClient({
        'frataxin': make_match(alias='frataxin', cui='C1414812', name='FXN gene', score=0.0175),
        'ftx': make_match(alias='ftx', cui='C3147341', name='FTX gene', score=0.6152),
    })
    selected = select_best_umls_api_match(
        aliases, client, canonical_type='genetic_factor', alias_provenance=provenance
    )
    assert selected is not None
    assert selected.cui == 'C1414812'
    assert selected.alias == 'frataxin'


def test_trace_path_uses_same_ftx_selection_policy():
    aliases = ['frataxin', 'ftx']
    provenance = [primary_provenance('frataxin', 'FTX'), concept_name_provenance('ftx')]
    client = FakeClient({
        'frataxin': make_match(alias='frataxin', cui='C1414812', name='FXN gene', score=0.0175),
        'ftx': make_match(alias='ftx', cui='C3147341', name='FTX gene', score=0.6152),
    })
    selected, trace = select_best_umls_api_match_with_trace(
        aliases, client, canonical_type='genetic_factor', alias_provenance=provenance
    )
    assert selected is not None
    assert selected.cui == 'C1414812'
    rows = [row for row in trace if row.get('selected_final')]
    assert len(rows) == 1
    assert rows[0]['query_alias'] == 'frataxin'


def test_dmd_incompatible_primary_long_form_cannot_override_gene_symbol():
    aliases = ['duchenne muscular dystrophy', 'dmd']
    provenance = [primary_provenance('duchenne muscular dystrophy', 'DMD'), concept_name_provenance('dmd')]
    client = FakeClient({
        'duchenne muscular dystrophy': make_match(
            alias='duchenne muscular dystrophy', cui='C0013264',
            name='Muscular Dystrophy, Duchenne', score=1.0, compatible=False
        ),
        'dmd': make_match(alias='dmd', cui='C1414083', name='DMD gene', score=0.6152),
    })
    selected = select_best_umls_api_match(
        aliases, client, canonical_type='genetic_factor', alias_provenance=provenance
    )
    assert selected is not None
    assert selected.cui == 'C1414083'


def test_weak_primary_long_form_does_not_gain_priority():
    aliases = ['example expanded form', 'abc']
    provenance = [primary_provenance('example expanded form', 'ABC'), concept_name_provenance('abc')]
    client = FakeClient({
        'example expanded form': make_match(
            alias='example expanded form', cui='C_LONG', name='Example expanded gene',
            score=0.40, synonym_supported=False
        ),
        'abc': make_match(
            alias='abc', cui='C_SHORT', name='ABC gene', score=0.90,
            synonym_supported=False
        ),
    })
    selected = select_best_umls_api_match(
        aliases, client, canonical_type='genetic_factor', alias_provenance=provenance
    )
    assert selected is not None
    assert selected.cui == 'C_SHORT'
