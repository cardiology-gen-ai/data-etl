# Knowledge Graph Pipeline — Cardiology ESC Guidelines

End-to-end pipeline that turns ESC clinical practice guidelines (PDF) into
a Neo4j knowledge graph combining narrative sections, prose-level
biomedical concepts, structured clinical recommendations and UMLS-linked
identifiers.

The pipeline has two halves with a clean handover via disk artifacts:

- **Preprocessing** — PDF → markdown → TOC → sections → tables → chunks →
  extracted recommendations → acronyms. Owned by the preprocessing
  managers (`MarkdownConverter`, `TOCExtractionManager`,
  `SectionAttributionManager`, `HierarchicalChunkingManager`,
  `RecommendationExtractionManager`, `SectionLinkingManager`).
  Produces files on disk. Does **not** touch Neo4j.
- **KG build** — artifacts → Neo4j. Loads chunks as `Section` nodes,
  runs the prose entity extractor, writes the structured recommendation
  layer, optionally writes recommendation→section links, then runs the
  global passes (disambiguation, UMLS normalization, embeddings, sanity
  checks). Owned by `build_kg.py`.

The two halves communicate via files only. Preprocessing is slow,
deterministic, and expensive to redo; KG build is fast, idempotent, and
will be rerun many times while tuning prompts, schemas, or constraints.

---

## Table of contents

1. [Architecture](#architecture)
2. [Prerequisites and environment variables](#prerequisites-and-environment-variables)
3. [On-disk artifacts](#on-disk-artifacts)
4. [The three LLM models](#the-three-llm-models)
5. [Pipeline stages](#pipeline-stages)
6. [Graph schema](#graph-schema)
7. [Entity schema and concept namespace](#entity-schema-and-concept-namespace)
8. [Acronym handling](#acronym-handling)
9. [Embeddings: storage policy](#embeddings-storage-policy)
10. [Rate-limit handling and retry pass](#rate-limit-handling-and-retry-pass)
11. [Running the pipeline](#running-the-pipeline)
12. [Idempotency and reruns](#idempotency-and-reruns)
13. [Configuration knobs](#configuration-knobs)
14. [Sanity checks](#sanity-checks)
15. [Troubleshooting](#troubleshooting)
16. [Extension points](#extension-points)
17. [Glossary](#glossary)

---

## Architecture

```
                            +-----------------------+
                            | PDF (ESC guideline)   |
                            +-----------+-----------+
                                        |
        +-------------------------------+-------------------------------+
        |                  PREPROCESSING (no Neo4j)                     |
        |                                                               |
        |  MarkdownConverter            -> markdown + page anchors      |
        |  TOCExtractionManager          -> TOC tree (json)              |
        |  SectionAttributionManager     -> tables attributed to sections|
        |  HierarchicalChunkingManager   -> chunks pickle per document   |
        |  RecommendationExtractionMgr   -> recommendations JSON         |
        |  SectionLinkingManager         -> recommendation->section json |
        |  (acronyms cached per doc)                                    |
        +-------------------------------+-------------------------------+
                                        |
                                        | (files on disk)
                                        v
        +-------------------------------+-------------------------------+
        |                          KG BUILD                             |
        |                                                               |
        |  Per-document, in order:                                      |
        |   1) graph_loader   : chunks -> Document + Section nodes      |
        |   2) add_entities   : prose LLM -> :Concept + :MENTIONS       |
        |   3) add_recs       : structured -> :Recommendation + roles   |
        |   4) section_linking: rec->section override edges (optional)  |
        |                                                               |
        |  Global, once:                                                |
        |   5) disambiguate_concepts: canonical_type by majority        |
        |   6) umls_normalization   : scispacy + UMLS REST API hybrid   |
        |   7) add_embeddings       : section vectors                   |
        |   8) sanity_checks        : invariants and audit              |
        +---------------------------------------------------------------+
```

---

## Prerequisites and environment variables

### Required environment variables

Discover the full list by running this against your `llm_utils.py`:

```bash
grep -n "_get_required_env" knowledge_graph/llm_utils.py
```

Every name appearing in those calls must be set, otherwise the pipeline
will raise `RuntimeError: Missing required environment variable: ...`
at first LLM call. The minimal set in the current codebase:

```bash
# Neo4j Aura
NEO4J_URI=neo4j+s://<your-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<aura-password>

# Chat LLM (used by add_entities AND RecommendationExtractionManager)
KG_CHAT_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# UMLS REST API (only if umls_mode in {"api", "hybrid"})
UMLS_API_KEY=...
```

Other env vars likely required by `llm_utils._get_required_env`
(verify against the actual `grep` output):

```bash
KG_CHAT_PROVIDER=openai            # provider for chat model
KG_EMBEDDING_MODEL=text-embedding-3-small
KG_EMBEDDING_PROVIDER=openai
```

Load them at the top of your entry script:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Python packages

The KG-side modules depend on:

- `neo4j` — Neo4j Python driver
- `openai` — chat + embeddings client (used both directly and via
  `langchain-openai`)
- `langchain-core`, `langchain-openai` — structured-output for the
  recommendation extractor
- `pydantic` — schemas
- `scispacy` + a model (e.g. `en_core_sci_scibert`) — local UMLS linker
- `requests` — UMLS REST API client
- `rapidfuzz` — fuzzy matching for the acronym helpers (transitive,
  used by `validate_entities`)

scispaCy models are installed separately. The default
`en_core_sci_scibert` is around 1 GB and needs to be present in the
scispaCy cache before the UMLS step runs.

---

## On-disk artifacts

After preprocessing for a `doc_id`, this is the on-disk layout the KG
build expects:

```
<preprocessing_output>/                       # rendered_conversion_config.output_folder.folder
├── <doc_id>.md                               # markdown rendering of the PDF
├── <doc_id>_tables/                          # extracted tables
│   └── ...
└── abbreviations/
    └── <doc_id>.json                         # {short_form: long_form}

<chunks_folder>[_<header_levels>]/            # HierarchicalChunkingManager output
└── <doc_id>.pkl                              # list[langchain Document]

<kg_folder>/
├── recommendations/
│   └── <doc_id>_recommendations.json         # list[ExtractionEntry]
├── section_linking/
│   └── <doc_id>_section_links.json           # list[SectionLink] (optional)
├── entity_review/                            # JSONL audit written by stage 2
│   ├── <doc_id>_accepted.jsonl
│   ├── <doc_id>_rejected.jsonl
│   └── <doc_id>_summary.json
└── umls_review/                              # JSONL audit from umls_normalization
    └── ...
```

The `chunks_folder` value is the chunker's
`config.chunks_folder.folder`. When the chunker is built with
`header_levels > 0`, the actual folder is suffixed
(`<chunks_folder>_<header_levels>`); the `KGPaths` dataclass needs the
folder with the suffix already applied.

---

## The three LLM models

There are three LLM-like models in play, each configured in its own
place. They are independent and can be different.

| stage | what it does | how it's configured |
| --- | --- | --- |
| **3. Recommendation extraction** | Structured output (Pydantic) from each recommendation table row | `RecommendationExtractionManager(model="gpt-4o", ...)` — explicit constructor argument. Stored on each `ExtractionEntry.model`. |
| **2. Prose entity extraction** | Concept extraction from section text | Reads `KG_CHAT_MODEL` env var via `llm_utils.get_chat_model_name()`. Cannot be overridden by kwargs. |
| **7. Section embeddings** | Vector embedding of each Section's text | `add_embeddings_to_sections(..., embedding_model="text-embedding-3-small", ...)` — explicit. Falls back to `KG_EMBEDDING_MODEL` when None. |

The fourth player is the **scispaCy model** used by stage 6 for UMLS
linking. Not an LLM strictly speaking. Configured via
`umls_scispacy_kwargs={"model_name": "en_core_sci_scibert", ...}`.

If you want stages 2 and 3 to share the same LLM, set the env var and
read it back when constructing the recommendation manager:

```python
import os
KG_CHAT_MODEL = os.environ["KG_CHAT_MODEL"]

rec_extractor = RecommendationExtractionManager(
    output_folder=kg_folder / "recommendations",
    tabs_folder=preprocessing_output,
    app_id=app_id,
    model=KG_CHAT_MODEL,
    acronym_folder=preprocessing_output / "abbreviations",
)
```

---

## Pipeline stages

### 1. Structural graph load (per document)

**Module:** `build_kg._write_chunks_to_graph` (wraps
`graph_loader.py` low-level helpers).
**Input:** `<doc_id>.pkl` (list of langchain `Document`s).
**Output:**

- `(:Document {doc_id})`
- `(:Section {uid, doc_id, section_id, title, level, text, embed,
  page_start, page_end, parent_section_id, section_type,
  breadcrumb_path, chunk_id, ...})` — one per chunk.
- `(:Document)-[:HAS_SECTION]->(:Section)`
- `(:Section)-[:HAS_CHILD]->(:Section)` — TOC hierarchy.
- `(:Section)-[:NEXT]->(:Section)` — reading order.

The chunker's metadata (`TOCChunkMetadata`) is mapped to the dict shape
expected by `graph_loader.normalize_section_record`:

| TOCChunkMetadata          | normalize_section_record |
| ------------------------- | ------------------------ |
| `filename`                | `doc_id`                 |
| `section_id`              | `section_id`             |
| `section_title`           | `section_title` → `title` |
| `section_level`           | `section_level` → `level` |
| `page_content`            | `text` (breadcrumb-prepended when header_levels > 0) |
| `is_empty`, `embed`       | unchanged                |
| `page_start`/`page_end`   | unchanged                |
| `parent_section_id`       | unchanged                |
| `section_type`            | extra property            |
| `headers` (flattened to "A > B > C") | `breadcrumb_path` (extra property) |
| `chunk_id`                | extra property            |

The `embed` flag is decided here using `infer_should_embed`: explicit
`embed` value wins, otherwise empty sections and sections shorter than
`MIN_TEXT_CHARS_TO_EMBED` (default 20) are excluded.

`replace_existing_document=True` (default) detach-deletes the document
before reloading. Use this for the typical rerun-while-tuning case.

#### Caveat on missing parents

The chunker may not emit chunks for sections that have no own body
text (e.g. a level-1 section that's just a container of level-2
subsections, with `header_levels=2`). Their level-2 children still
carry `parent_section_id` pointing to that container, but the
container has no node in the graph; the `HAS_CHILD` MERGE silently
skips. The sanity check `recommendation_without_containing_section`
flags any side effects on recommendations.

### 2. Prose entity extraction (per document)

**Module:** `add_entities.add_entities_from_sections`
**Input:** the `Section` nodes from stage 1; the per-doc acronym JSON.
**Output:** `(:Section)-[:MENTIONS {observed_types, raw_name, raw_type,
support_method, matched_text, source, ...}]->(:Concept)` edges. Concept
nodes are MERGEd on normalized name (`entity_schema.normalize_name`).

For each eligible section, the LLM extracts a list of `{name, type}`
concepts. The validator
(`validate_entities.validate_concepts_against_source`) accepts only
concepts whose name is supported by the section source text — soft
substring match tolerant of plurals, hyphens, spelling variants, and
acronym expansion when an acronym dict is available.

Multiple sections may contribute multiple `observed_types` to the same
`:Concept`; the final type is decided in stage 5 by majority of support.

Audit trail (JSONL) is written to `entity_review/<doc_id>_accepted.jsonl`
and `entity_review/<doc_id>_rejected.jsonl`.

**Valid `add_entities_kwargs`** you can pass through `build_kg`:

| kwarg                                  | default     | purpose |
| -------------------------------------- | ----------- | ------- |
| `use_section_text`                     | `False`     | extract from section body, not only the title |
| `max_sections`                         | `None`      | cap for debug runs |
| `max_sections_per_batch`               | `2`         | sections per LLM call |
| `max_batch_chars`                      | `12000`     | character cap per batch |
| `emergency_max_single_chars`           | `12000`     | safety truncation |
| `skip_processed`                       | `True`      | resume support |
| `replace_section_mentions`             | `True`      | wipe MENTIONS before rewriting |
| `export_entity_review`                 | `True`      | dump accepted/rejected JSONL |
| `clear_previous_entity_review`         | `True`      | clear old JSONLs at start |
| `include_source_preview_in_review`     | `False`     | bigger audit records |
| `use_acronym_validation`               | `True`      | enable acronym-aware validation |

`driver`, `doc_id`, `entity_review_output_dir`, `acronym_dir` are
already supplied by `build_kg` — don't duplicate.

`model_name` is **not** a parameter: the chat model comes from
`KG_CHAT_MODEL`.

### 3. Structured recommendation writer (per document)

**Module:** `add_recommendations.add_recommendations_from_extractions`
**Input:** `<doc_id>_recommendations.json` (produced upstream by
`RecommendationExtractionManager`, v2.0 schema with typed `EntityTerm`).
**Output:**

- `(:Recommendation {uid, recommendation_id, doc_id, table_id,
  row_index, table_caption, section_path, container_id, source_text,
  group_header, effective_source, modality, class, level,
  logical_operator, extraction_notes, prompt_version, model,
  extracted_at, validation_flags})`
- Best-effort fallback edge `(:Section)-[:CONTAINS_RECOMMENDATION]->(:Recommendation)`
  using `container_id` (overridden by stage 4 when section_linking is
  available — see below).
- Role-typed edges to the shared `:Concept` namespace:

| Source role          | Relationship type        | Notes |
| -------------------- | ------------------------ | ----- |
| `requires`           | `HAS_INDICATION`         | Inclusion criterion |
| `excludes`           | `HAS_CONTRAINDICATION`   | Exclusion criterion |
| `intervention`       | `RECOMMENDS_ACTION`      | `role=intervention` |
| `alternative`        | `RECOMMENDS_ACTION`      | `role=alternative`, `role_index>0` |
| `purpose`            | `HAS_PURPOSE`            | Outcome / aim |

Each role edge carries:

- `role`, `role_index`
- flat qualifier properties (`qualifier_severity`, `qualifier_dose`,
  `qualifier_threshold`, `qualifier_negated`, `qualifier_route`, …)
- `raw_name`, `raw_type` (the LLM's literal output before
  normalization)
- `expected_semgroups` — the prior set of UMLS Semantic Groups
  acceptable for that role (used by stage 6 as a tiebreaker)
- validation evidence (`support_method`, `matched_text`,
  `acronym_short`, `acronym_definition`) when produced by
  `validate_concepts_against_source`.

Each role-extracted concept is also written as
`(:Section)-[:MENTIONS {source: 'recommendation'}]->(:Concept)` so the
global disambiguator counts these mentions uniformly with the prose
branch.

### 4. Section linking (optional, per document)

**Module:** `build_kg._process_document_section_links`.
**Input:** `<doc_id>_section_links.json` produced by
`SectionLinkingManager`.
**Output:** authoritative `(:Section)-[:CONTAINS_RECOMMENDATION
{match_strategy, target_section_id, target_section_title,
linking_version, linked_at}]->(:Recommendation)` edges.

`SectionLinkingManager` does **recommendation → section(s)** mapping
using three strategies, in order:

1. `section_id_exact` — the recommendation row was attributed to a
   section whose numeric id matches a chunk exactly (e.g. `5.1.1`).
2. `section_id_subtree` — the attributed section has descendants in the
   chunks; the recommendation gets linked to all of them (e.g. a row
   under `5.1` ends up linked to `5.1.1`, `5.1.2`, ...).
3. `title_fallback` — the section id couldn't be extracted; matched by
   normalised title.

When the linker output is present, this stage:

- removes any existing `CONTAINS_RECOMMENDATION` edges to each
  affected Recommendation (including the heuristic ones from stage 3);
- creates new edges from every `chunk_section_ids[i]`;
- stores `match_strategy` on each edge so downstream queries can
  filter (e.g. exclude `title_fallback` when high precision is
  needed).

When the linker file is absent, this stage is a no-op and the stage-3
heuristic edges remain.

### 5. Concept disambiguation (global)

**Module:** `entity_disambiguation.disambiguate_concepts`
**Input:** all `:MENTIONS` edges (from stages 2 and 3).
**Output:** for each `:Concept`, the canonical type is decided by
majority of section-level support across the whole graph.

Possible `canonical_type` values after this stage:

- one of `entity_schema.ALLOWED_TYPES` — unique winner exists.
- `"ambiguous"` — top-supported types tied and no defensible
  pre-existing canonical can be kept.
- `"no_supported_type"` — concept has no current section-level support
  (orphan-ish).

Orphan concepts (no incoming MENTIONS) are deleted in this step.

Also sets `needs_type_review` (audit flag) and
`type_resolution_status`.

### 6. UMLS normalization (global)

**Module:** `umls_normalization.normalize_concepts_with_umls`
**Input:** all `:Concept` nodes in the graph.
**Output:** for each Concept, UMLS metadata when a confident match is
found: `cui`, `umls_canonical_name`, `umls_semantic_types`,
`umls_semantic_groups`, `umls_score`, `umls_source`. Optional
`(:Concept)-[:SAME_AS]->(:Concept)` edges for cross-concept identity
collapse.

The upstream module supports three backends:

- `SCISPACY_BACKEND = "scispacy"` — local scispaCy + KB.
- `UMLS_API_BACKEND = "umls_api"` — UMLS REST API.
- `FUZZY_ONLY_BACKEND = "fuzzy_only"` — degraded fuzzy duplicate
  detection only.

`build_kg` wraps these in four orchestration modes via `umls_mode`:

- `"scispacy"` — only local
- `"api"` — only REST API (needs `UMLS_API_KEY`)
- `"hybrid"` (default) — scispaCy first on all Concepts, then REST API
  on whatever didn't get a confident match. Each Concept ends up with
  `umls_source = "scispacy"` or `"api"`.
- `"fuzzy"` — degraded mode

**Role-aware reranking.** The `expected_semgroups` hint persisted on
role edges by stage 3 is *not* automatically read by upstream
`select_best_*` functions. To enable it, wire
`kg.umls_role_hints.aggregate_semgroup_priors_for_concept` +
`score_candidate_with_role_prior` into your local copy of
`umls_normalization.select_best_umls_match` /
`select_best_umls_api_match`. Without that wiring, the hints sit on
edges and are queryable but inert.

### 7. Section embeddings (global)

**Module:** `add_embeddings.add_embeddings_to_sections`
**Input:** `:Section` nodes with `embed=True` (and currently
`has_embedding=False` unless `force_reembed`).
**Output:** `embedding` (float list), `has_embedding=true`,
`embedding_model`, `embedding_dim`, `embedding_updated_at`,
`embedding_status` on each embedded Section.

Sections with no usable text are marked `embedding_status='skipped_empty'`
and have any stale vector removed. Sections that the embedding backend
fails on are marked `'failed'` with a timestamp.

The vector index is **not** created by this stage. Create it once after
the first successful run:

```cypher
CREATE VECTOR INDEX section_embedding IF NOT EXISTS
FOR (s:Section) ON (s.embedding)
OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
}}
```

(Adjust `dimensions` to your model. 1536 = `text-embedding-3-small`.)

### 8. Sanity checks (global)

**Module:** `sanity_checks.run_sanity_checks` plus
`sanity_checks_recommendations.RECOMMENDATION_CHECKS`.

Cypher checks grouped by phase (`structure`, `entities`, `embeddings`,
`recommendations`, `full`). Each returns a count or sample with a
severity (INFO / WARNING / ERROR). Output is a structured dict you can
dump to JSON for CI.

Recommendation-specific checks include:

- modality / class / role-edge distributions
- modality↔class mismatches
- non-verbatim spans
- orphan recommendations (no role edges, or no Section attachment)
- concepts with `canonical_type='ambiguous'` referenced by recs
- concepts cited by recs without a UMLS match
- role edges whose `expected_semgroups` don't intersect the linked
  concept's actual `umls_semantic_groups`
- recommendation→section link strategy distribution
- recommendations linked via `title_fallback` only (low-precision)

---

## Graph schema

### Nodes

| Label             | Key            | Notable properties |
| ----------------- | -------------- | ------------------ |
| `Document`        | `doc_id`       | (one per PDF) |
| `Section`         | `uid`          | `doc_id`, `section_id`, `printed_section_id`, `title`, `level`, `text`, `is_empty`, `embed`, `page_start`, `page_end`, `part_index`, `part_count`, `quality_flags`, `boundary_source`, `section_type`, `breadcrumb_path`, `chunk_id`; pipeline state: `entity_extracted{,_at,_status,_failed_at}`, `embedding{,_model,_dim,_status,_updated_at,_failed_at}`, `has_embedding`. |
| `Concept`         | `name`         | `observed_types`, `canonical_type`, `type_resolution_status`, `needs_type_review`; after stage 6: `cui`, `umls_canonical_name`, `umls_semantic_types`, `umls_semantic_groups`, `umls_score`, `umls_source`. |
| `Recommendation`  | `uid`          | `recommendation_id`, `doc_id`, `table_id`, `row_index`, `table_caption`, `section_path`, `container_id`, `source_text`, `group_header`, `effective_source`, `class`, `level`, `modality`, `logical_operator`, `extraction_notes`, `prompt_version`, `model`, `extracted_at`, `validation_flags`. |

### Constraints / indexes

- `Document.doc_id` unique
- `Section.uid` unique
- `Concept.name` unique
- `Recommendation.uid` unique
- indexes on `Section.doc_id`, `Concept.canonical_type`,
  `Concept.type_resolution_status`, `Concept.needs_type_review`,
  `Recommendation.doc_id`, `Recommendation.modality`,
  `Recommendation.class`.

### Relationships

| Type                       | From → To                   | Notes |
| -------------------------- | --------------------------- | ----- |
| `HAS_SECTION`              | `Document → Section`        | Document membership. |
| `HAS_CHILD`                | `Section → Section`         | TOC hierarchy. |
| `NEXT`                     | `Section → Section`         | Reading order. |
| `MENTIONS`                 | `Section → Concept`         | Prose mention or recommendation-derived. Properties: `observed_types`, `raw_name`, `raw_type`, validation evidence, `source ∈ {"prose","recommendation"}`. |
| `CONTAINS_RECOMMENDATION`  | `Section → Recommendation`  | When written by stage 4: carries `match_strategy`, `target_section_id`, `target_section_title`, `linking_version`. |
| `HAS_INDICATION`           | `Recommendation → Concept`  | From `population.requires`. Carries qualifiers, `expected_semgroups`, `raw_name`, validation evidence. |
| `HAS_CONTRAINDICATION`     | `Recommendation → Concept`  | From `population.excludes`. |
| `RECOMMENDS_ACTION`        | `Recommendation → Concept`  | Intervention or alternative; carries `role`, `role_index`, dose/route, `expected_semgroups`. |
| `HAS_PURPOSE`              | `Recommendation → Concept`  | Clinical outcome / aim. |
| `SAME_AS`                  | `Concept → Concept`         | Optional, from UMLS normalization when two Concepts collapse to one CUI. |

### Example query

"All class-I recommendations for HFrEF involving a beta-blocker,
together with their section":

```cypher
MATCH (s:Section)-[:CONTAINS_RECOMMENDATION]->(r:Recommendation)
WHERE r.class = 'I'
MATCH (r)-[:HAS_INDICATION]->(:Concept {name: 'hfref'})
MATCH (r)-[:RECOMMENDS_ACTION]->(drug:Concept)
WHERE drug.canonical_type = 'drug_or_drug_class'
  AND 'CHEM' IN coalesce(drug.umls_semantic_groups, [])
RETURN s.title AS section, r.source_text AS recommendation, drug.name AS drug
```

---

## Entity schema and concept namespace

One `Concept` namespace, shared by the prose extractor (stage 2) and
the recommendation writer (stage 3). This is the central structural
decision of the design — without it you'd have two parallel graphs and
the join between them is the whole point of a KG.

Schema in `kg/entity_schema.py`:

```text
ALLOWED_TYPES = {
    disease, clinical_finding, risk_factor, genetic_factor,
    biomarker, diagnostic_test, imaging_modality, score_or_risk_model,
    drug_or_drug_class, procedure_or_intervention, device,
    complication_or_comorbidity, care_strategy, anatomical_structure,
    clinical_outcome,     # for recommendation purpose clauses
    dose_or_regimen,      # for dose entities (rare)
}
```

Both pipelines normalize names through `normalize_concept`
(lowercase + whitespace collapse + enclosing punctuation strip +
blocklist filter + type alias resolution). Two LLMs writing
"Atrial Fibrillation" and "atrial fibrillation," MERGE into the same
node.

Type ambiguity is preserved on `MENTIONS.observed_types`. The
canonical type is decided globally in stage 5 — *not* per document —
so the winner reflects consensus across the whole corpus.

The recommendation extractor uses the same `ALLOWED_TYPES` with a
strict Pydantic `Literal`, preventing the LLM from inventing types.

---

## Acronym handling

Per-document acronym dicts live at
`<preprocessing_output>/abbreviations/<doc_id>.json` as flat
`{short_form: long_form}` JSON objects, produced upstream.

They are consumed in two places:

- **Stage 2** (prose entity extraction):
  `add_entities_from_sections` is called with `acronym_dir` and
  `use_acronym_validation=True`. `validate_entities.py` then accepts a
  concept whose long form appears in the section via its acronym
  short form, and can expand a raw acronym in LLM output to its long
  form before graph writing.
- **Stage 3** (recommendation extraction): the
  `RecommendationExtractionManager` loads
  `abbreviations/<doc_id>.json` and persists it on each
  `ExtractionEntry.acronyms_snapshot`. `add_recommendations.py` passes
  it to `validate_concepts_against_source` so the recommendation-level
  entities benefit from the same soft validation.
- **Stage 6** (UMLS normalization): the same `acronym_dir` is passed
  through for alias expansion during linking.

No extraction step is needed in the KG build: the dicts are produced
once during preprocessing.

---

## Embeddings: storage policy

**Default:** embeddings are stored on the `Section` node, inside Neo4j.

Why (for this corpus):

- The Aura native vector index lets you combine vector similarity with
  graph traversal in one Cypher query — important for RAG with
  graph-derived constraints ("sections similar to X that live inside
  a class-I recommendation about anticoagulation").
- Single source of truth for backups, ACLs, rebuilds.
- A 1536-d float32 vector costs ~6 KB; even thousands of sections fit
  comfortably in any paid Aura tier.

When to revisit:

- If you scale to thousands of documents, embeddings start to dominate
  Neo4j's footprint. At that point move them out: keep
  `embedding_model`, `embedding_dim`, `embedding_status` as stubs on
  `Section` and put vectors in pgvector / Qdrant / Weaviate, joined
  back by `Section.uid`. The pipeline already has the right
  abstraction — `add_embeddings_to_sections` would just be
  reimplemented.
- If you want to A/B compare embedding models you'll need either an
  extra slot per model (cheap but ugly) or a side store.

Hygiene now:

- Keep `MIN_TEXT_CHARS_TO_EMBED` ≥ 20 so headers and one-line stubs
  don't get embedded (`embed=false` saves cost and reduces retrieval
  noise).
- Run `check_embeddings.py` after stage 7 to confirm coverage and
  consistency.

Not recommended:

- Embedding `Recommendation` nodes — text is too short, and the role
  edges already carry richer structure. Embed the `Section` and
  traverse `:CONTAINS_RECOMMENDATION`.
- Embedding `Concept` nodes — they are normalized names; if you want
  concept-level vectors, use scispaCy KB embeddings via the linked
  CUI.

---

## Rate-limit handling and retry pass

LLM calls in stages 2 and 3 hit OpenAI rate limits in two ways:
**RPM** (requests per minute) and **TPM** (tokens per minute). The
recommendation extractor handles both with a dedicated retry strategy.

### Per-row retry (during extraction)

`ExtractionsManager._invoke_chain_with_retry` wraps every
`chain.invoke` call:

- catches `RateLimitError`, `APITimeoutError`, `APIConnectionError`,
  `InternalServerError`
- honors `Retry-After` header when present
- otherwise applies capped exponential backoff with full jitter
- logs every attempt with exception type, message, wait, source
- after `max_retries` attempts the row is marked `ok=False` and the
  loop continues to the next row

OpenAI's own client retries are disabled (`max_retries=0` in
`ChatOpenAI`) so all retries flow through the manager and are visible
in logs.

### Optional cleanup pass (end of extraction)

After the main loop, the manager can run a second pass on entries that
are still `ok=False` *and* whose `error` belongs to a retryable class.
The pass:

- cools down for `retry_failed_cooldown` seconds (default 30) to let
  the provider's per-minute window reset
- runs only on retryable failures (`RateLimitError` etc.) — never on
  `ValidationError` or other non-transient errors
- replaces successful retries in-place in the catalog (no append at
  the tail)
- saves after each successful recovery

Configured at construction time:

```python
RecommendationExtractionManager(
    ...,
    request_timeout=60.0,
    max_retries=6,
    initial_backoff=2.0,
    max_backoff=60.0,
    inter_request_delay=0.0,           # pause between successive rows

    retry_failed_pass=True,            # enable the cleanup pass
    retry_failed_cooldown=30.0,
    retry_failed_max_retries=None,     # None = same as max_retries
    retry_failed_inter_request_delay=1.0,
)
```

### Standalone retry on existing catalog

To re-run the cleanup pass later on a JSON catalog already on disk
(without re-extracting anything), call `retry_failed_file`:

```python
stats = rec_extractor.retry_failed_file(
    pathlib.Path("ESC_HF_2023.pdf"),
    cooldown=0,                        # default 0 because you've been waiting
    max_retries_per_row=8,
    inter_request_delay=1.5,
)
# {'candidates': 12, 'recovered': 11, 'still_failing': 1, 'unrecoverable': 0}
```

Useful for batch operations:

```python
for f in input_files:
    try:
        rec_extractor.retry_failed_file(pathlib.Path(f))
    except FileNotFoundError:
        # never extracted in the first place
        continue
```

The method is idempotent: a rerun on a clean catalog returns
`{'candidates': 0, ...}` immediately.

### What never gets retried

- `ValidationError` from Pydantic — the LLM produced output that
  doesn't satisfy the schema. Fix prompt or schema, not by retrying.
- Malformed JSON from the LLM — same.
- Anything that doesn't match the retryable class set (no `RateLimit`,
  `Timeout`, `Connection`, `InternalServer`, or 5xx prefix).

---

## Running the pipeline

### End-to-end build

```python
import os
import pathlib
from dotenv import load_dotenv
from neo4j import GraphDatabase

from kg.build_kg import build_kg, KGPaths
from kg.recommendation_extraction_manager import RecommendationExtractionManager

load_dotenv()

# 1. Run preprocessing managers first (not shown — produces the artifacts)

# 2. Make sure the recommendations JSON is on disk for every doc
rec_extractor = RecommendationExtractionManager(
    output_folder=kg_folder / "recommendations",
    tabs_folder=rendered_conversion_config.output_folder.folder,
    app_id=app_id,
    model=os.environ["KG_CHAT_MODEL"],
    acronym_folder=rendered_conversion_config.output_folder.folder / "abbreviations",
    request_timeout=60.0,
    max_retries=6,
    inter_request_delay=0.5,
)
for f in input_files:
    rec_extractor(pathlib.Path(f))

# 3. Open Neo4j and build the graph
driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
)
driver.verify_connectivity()

paths = KGPaths(
    preprocessing_output=rendered_conversion_config.output_folder.folder,
    kg_folder=kg_folder,
    chunks_folder=pathlib.Path(
        str(chunker.config.chunks_folder.folder)
        + (f"_{chunker.header_levels}" if chunker.header_levels > 0 else "")
    ),
)

doc_ids = [pathlib.Path(f).stem for f in input_files]

try:
    summary = build_kg(
        driver=driver,
        paths=paths,
        doc_ids=doc_ids,
        # stage 2: tune throughput vs. rate limits
        add_entities_kwargs={
            "use_section_text": True,
            "max_sections_per_batch": 1,
            "max_batch_chars": 6000,
        },
        # stage 6: hybrid scispaCy + REST API
        umls_mode="hybrid",
        umls_scispacy_kwargs={"model_name": "en_core_sci_scibert", "threshold": 0.85},
        umls_api_kwargs={"threshold": 0.80},
        # stage 7: embeddings
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
    )
    print(summary)
finally:
    driver.close()
```

### Single document

```python
from kg.build_kg import process_document

process_document(driver, paths, doc_id="2024_esc_hf")
```

Then run the global passes manually once you've ingested all docs:

```python
from kg.build_kg import (
    _run_disambiguation,
    _run_umls_normalization,
    _run_embeddings,
    _run_sanity_checks,
)
_run_disambiguation(driver)
_run_umls_normalization(driver, paths, mode="hybrid")
_run_embeddings(driver, doc_ids=[...])
_run_sanity_checks(driver, mode="full")
```

### Selective rerun (changed prompt only)

```python
build_kg(
    driver=driver, paths=paths, doc_ids=doc_ids,
    replace_existing_document=False,    # keep Section nodes
    skip_section_links=True,            # no need
    skip_embeddings=True,                # no need
    # rerun stages 2 + 3 + 5 + 6 implicitly (defaults)
)
```

---

## Idempotency and reruns

| change | stages to rerun | flags |
| ------ | --------------- | ----- |
| New PDF | all | `build_kg` with the new doc_id |
| Re-chunked existing PDF | 1 + downstream | `replace_existing_document=True` |
| Tuned prose-entity prompt | 2 + 5 + 6 | `replace_existing_document=False` + `skip_recommendations=True` + `skip_embeddings=True` |
| Bumped `PROMPT_VERSION` in recommendation extractor | 3 + 5 + 6 | upstream `RecommendationExtractionManager` auto-rextracts only rows with stale prompt; then rerun stages 3+5+6 |
| Added a new `ALLOWED_TYPES` value | 2 + 3 + 5 | full rerun of those stages |
| Changed embedding model | 7 | `embedding_kwargs={"force_reembed": True}`; recreate vector index with new dim |
| Just want to retry rate-limit failures | none (LLM only) | call `rec_extractor.retry_failed_file(path)` then rerun stage 3 |

Per-stage idempotency notes:

- **Stage 1** detach-deletes the document when
  `replace_existing_document=True`; orphan Concepts cleaned up; old
  Recommendations lose `CONTAINS_RECOMMENDATION` but the nodes
  themselves persist until stage 3 detach-deletes and rewrites them.
- **Stage 2** clears existing `MENTIONS` from the section it's
  processing (`replace_section_mentions=True`).
- **Stage 3** detach-deletes each Recommendation before rewriting; no
  stale edges accumulate.
- **Stage 4** drops all `CONTAINS_RECOMMENDATION` to affected recs and
  rewrites from authoritative source.
- **Stage 5** rebuilds `canonical_type` from scratch each run.
- **Stage 6** skips Concepts that already have a CUI unless
  `force_relink=True` (implementation-dependent).
- **Stage 7** skips Sections with `has_embedding=true` unless
  `force_reembed=True`.

---

## Configuration knobs

### Structural

- `MIN_TEXT_CHARS_TO_EMBED` (env, default 20) — minimum text length
  for embed eligibility.
- `replace_existing_document` (default `True`) — rerun semantics.

### Prose entity extraction

See `add_entities_kwargs` table above. Two knobs to know:

- `max_sections_per_batch` lower → more LLM calls, smaller payloads,
  more resilient to TPM caps.
- `use_section_text=False` (default) means title-only extraction —
  cheaper but lower recall. Set to `True` for full coverage.

### Recommendation extraction

- `model` (constructor) — chat model name (e.g. `gpt-4o-mini`).
- `temperature` (constructor) — default 0.0.
- `request_timeout`, `max_retries`, `initial_backoff`, `max_backoff`,
  `inter_request_delay` — retry/pacing.
- `retry_failed_pass`, `retry_failed_cooldown`,
  `retry_failed_max_retries`, `retry_failed_inter_request_delay` —
  optional cleanup pass after the main loop.

### UMLS

- `umls_mode` ∈ `{"scispacy", "api", "hybrid", "fuzzy"}` (default
  `"hybrid"`).
- `umls_scispacy_kwargs={"model_name": ..., "threshold": ...,
  "max_candidates": ...}`.
- `umls_api_kwargs={"threshold": ...,
  "api_timeout": ..., "api_rate_limit_per_second": ...}`.
- `umls_use_acronyms` (default `True`).

### Embeddings

- `embedding_provider`, `embedding_model`, `embedding_dimensions` —
  explicit override of the `KG_EMBEDDING_MODEL` env var.
- `embedding_batch_size` (default 8).
- `embedding_kwargs={"force_reembed": True, "max_chars_per_section":
  8000, ...}`.

### Sanity

- `sanity_check_mode` ∈ `{"structure", "entities", "embeddings",
  "recommendations", "full"}`.

---

## Sanity checks

Two surfaces produced automatically:

- **JSONL audit** in `entity_review/` per document with the section
  context, raw and normalized names, validation reason
  (`support_method=substring | acronym_expansion | ...`), matched
  text and pattern. Read via `entity_review_exports.read_jsonl`.
- **Cypher sanity-checks** queries grouped by phase. Run
  `run_sanity_checks(driver, mode="full")` for a structured report.

Manual probes you'll use often:

```cypher
-- Recommendations not attached to any section
MATCH (r:Recommendation)
WHERE NOT EXISTS { MATCH (:Section)-[:CONTAINS_RECOMMENDATION]->(r) }
RETURN count(r);

-- Concepts referenced by recs whose canonical type is still ambiguous
MATCH (r:Recommendation)-[]->(c:Concept {canonical_type: 'ambiguous'})
RETURN c.name, c.observed_types, count(r) AS recs
ORDER BY recs DESC LIMIT 20;

-- Role edges whose expected SemGroups don't match the linked Concept's UMLS
MATCH (:Recommendation)-[e]->(c:Concept)
WHERE e.expected_semgroups IS NOT NULL
  AND c.umls_semantic_groups IS NOT NULL
  AND none(sg IN c.umls_semantic_groups WHERE sg IN e.expected_semgroups)
RETURN type(e), e.role, c.name, c.umls_semantic_groups, e.expected_semgroups
LIMIT 20;

-- How heavily we're relying on title-fallback for rec->section linking
MATCH (:Section)-[e:CONTAINS_RECOMMENDATION]->(:Recommendation)
RETURN coalesce(e.match_strategy, '(heuristic)') AS strategy, count(*) AS c
ORDER BY c DESC;
```

---

## Troubleshooting

**`RuntimeError: Missing required environment variable: KG_CHAT_MODEL`**
Set `KG_CHAT_MODEL` (and possibly `KG_EMBEDDING_MODEL`,
`KG_CHAT_PROVIDER`, `KG_EMBEDDING_PROVIDER`) in your `.env`. Discover
all required env vars with
`grep -n "_get_required_env" knowledge_graph/llm_utils.py`.

**`TypeError: add_entities_from_sections() got an unexpected keyword
argument 'model_name'`**
The chat model is *not* a kwarg of that function — it's read from
`KG_CHAT_MODEL`. Remove `model_name` from `add_entities_kwargs`.

**`FileNotFoundError: Missing chunks pickle for <doc_id>`**
The chunker hasn't run, or `KGPaths.chunks_folder` points to the wrong
directory. Remember the chunker uses
`<chunks_folder>_<header_levels>` when `header_levels > 0`.

**`Recommendation` nodes written but never attached to a Section.**
The fallback `_section_uid(doc_id, container_id)` in
`add_recommendations.py` assumes
`Section.uid = "{doc_id}::{container_id}"`. If your `graph_loader`
follows a different scheme, override that helper. The check
`recommendation_without_containing_section` flags it. Also: stage 4
(section_linking) overrides this — verify the linker output exists
at `kg_folder/section_linking/<doc_id>_section_links.json`.

**Many `non_verbatim_*` flags on otherwise plausible recommendations.**
The LLM is paraphrasing. Either bump the few-shot examples in
`structured_recommendation.SYSTEM_PROMPT`, or accept that some
paraphrasing is OK and rely on the soft validation in stage 3.

**Many `canonical_type='ambiguous'` after stage 5.**
Two pipelines voting differently on the same Concept. Inspect
`observed_types` of the affected Concepts and consider adding type
aliases in `entity_schema._RAW_TYPE_ALIASES`.

**UMLS linker assigns DISO to an `intervention` Concept.**
The role hints are written but not consumed. Wire
`kg.umls_role_hints.aggregate_semgroup_priors_for_concept` +
`score_candidate_with_role_prior` into
`umls_normalization.select_best_*` and rerun stage 6. Sanity check
`recommendation_role_semgroup_hint_mismatch` reports these.

**Aura rejects embedding write with "vector dimension mismatch".**
You changed `embedding_model` mid-corpus without
`embedding_kwargs={"force_reembed": True}`. Force-reembed once, or
drop all `embedding`/`has_embedding` properties first and recreate
the vector index.

**Frequent `RateLimitError` even after retries.**
- Increase `inter_request_delay` (e.g. 1.0 s)
- Decrease `add_entities_kwargs={"max_sections_per_batch": 1}`
- Increase `max_retries`
- After a batch finishes with `ok=False` survivors, run
  `rec_extractor.retry_failed_file(path)` after the rate-limit window
  has reset.
- Long term: upgrade OpenAI tier.

**`No extractions catalog on disk` from `retry_failed_file`.**
Run `rec_extractor(filepath)` first; `retry_failed_file` only operates
on existing catalogs.

---

## Extension points

- **Add an entity type.** Edit `entity_schema.ALLOWED_TYPES` and the
  aliases. Reflect it in the prompt glossary of
  `structured_recommendation.SYSTEM_PROMPT` if relevant. Rerun stages
  2 + 3 + 5.

- **Add a recommendation role.** Add a field to
  `RecommendationStructure` (e.g. `monitoring: Optional[EntityTerm]`),
  extend `iter_entity_terms`, add a mapping in
  `add_recommendations._ROLE_TO_RELTYPE`, add the role to
  `umls_semantics.ROLE_TO_ACCEPTED_GROUPS`. Bump `PROMPT_VERSION` to
  trigger re-extraction.

- **Switch embedding model.** Set new `embedding_model` and matching
  `embedding_dimensions`. Force reembed once. Drop and recreate the
  vector index with the new dimensions.

- **Move embeddings out of Neo4j.** Replace
  `add_embeddings_to_sections` with a writer for your external store,
  leaving only `embedding_status`/`embedding_model`/`embedding_dim` on
  `Section`. Update RAG retrieval to query externally and join back by
  `Section.uid`.

- **Use a non-OpenAI LLM.** For stage 3, modify
  `structured_recommendation.build_extraction_chain` to construct the
  right langchain client (`ChatAnthropic`, `AzureChatOpenAI`,
  `ChatOllama`). For stage 2, the provider is decided by
  `llm_utils.get_chat_model_name` and its provider counterpart —
  extend `llm_utils.py`.

- **Cross-document Concept consolidation by alias.** The current
  pipeline merges by normalized name only. To pre-collapse by alias
  before UMLS, add a step between 5 and 6 that runs a curated alias
  dictionary over Concepts.

- **Wire role hints into UMLS reranking.** See
  `umls_role_hints.py` docstrings for the three helpers
  (`expected_semgroups_for_role`,
  `aggregate_semgroup_priors_for_concept`,
  `score_candidate_with_role_prior`) and patch
  `umls_normalization.select_best_*` to use them.

---

## Glossary

- **`doc_id`** — stem of the source PDF filename, used as a stable
  document identifier across the entire pipeline.
- **`Section.uid`** — globally unique section identifier,
  `"{doc_id}::{section_id}"`.
- **`Recommendation.uid`** — globally unique recommendation identifier,
  `"{doc_id}::{recommendation_id}"` with
  `recommendation_id = "{table_id}::row_{row_index}"`.
- **TUI** — UMLS Semantic Type identifier (~127 fine-grained
  categories).
- **Semantic Group (SemGroup)** — NLM coarse grouping of TUIs (15
  macro-categories: DISO, CHEM, ANAT, PROC, DEVI, PHYS, …).
- **`expected_semgroups`** — prior set of acceptable SemGroups for a
  role-typed edge, derived from
  `umls_semantics.ROLE_TO_ACCEPTED_GROUPS`.
- **`observed_types`** — list of local entity-schema types observed on
  a Concept across all its incoming MENTIONS edges.
- **`canonical_type`** — single `ALLOWED_TYPES` value resolved by the
  disambiguator; or `"ambiguous"` / `"no_supported_type"`.
- **Verbatim contract** — the LLM must copy entity spans literally
  from the source text. Enforced at prompt level and audited by
  `validate_extraction`; tolerated softly by
  `validate_concepts_against_source` (plurals, hyphens, spelling,
  acronyms).
- **Group header** — text of a section-row inside a recommendation
  table, propagated by `TableManager` to every row beneath it and
  prepended to the row text before the LLM sees it (yielding
  `effective_source`).
- **`match_strategy`** — how `SectionLinkingManager` mapped a
  recommendation to its section(s): `section_id_exact`,
  `section_id_subtree`, `title_fallback`, or `empty`.
- **`umls_source`** — for a UMLS-linked Concept, indicates which
  backend produced the CUI: `"scispacy"` (local) or `"api"` (UMLS
  REST API).
- **Effective source** — `group_header + recommendation_text`, the
  string actually shown to the LLM during recommendation extraction.