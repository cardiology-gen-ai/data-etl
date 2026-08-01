## How to run 

> ![NOTE]
> To run the code the module of the repo cardiology-gen-ai MUST be installed e.g. via
> ```
> uv pip install -e ../cardiology-gen-ai
> ```
> (change `../cardiology-gen-ai` with the relative path to the repo in your machine).

> Other dependencies can be installed using `uv pip install .`

The first step is to download the .pdf documents from [this](https://drive.google.com/drive/folders/1rgaemZ4Jetyz98ivTw8fpLIndgZ2jczn?usp=sharing) Google Drive folder and save them into the `data/pdfdocs` folder (you have to create it).

Then, start the python virtual environment with this command:
```
source .venv/bin/activate
```

Start the docker container:
```
docker compose up -d
```

Set the environmental variables:
```
export CONFIG_PATH="absolute/path/to/data-etl/config.json"
export APP_CONFIG_PATH="absolute/path/to/cardiology-gen-ai/config.json"
export QDRANT_URL="http://localhost:6333"
export INDEX_ROOT="absolute/path/to/data-etl"
```

Finally, start the main script:
```
uv run -m src.main
```

## Knowledge graph runs

For local runs on a laptop or workstation configure `data-etl/.env`
with the Neo4j target you want, then run the graph pipeline directly:

```bash
cd data-etl
KG_PIPELINE_PHASE=graph ../.venv/bin/python src/main_graph.py
KG_PIPELINE_PHASE=entities ../.venv/bin/python src/main_graph.py
KG_PIPELINE_PHASE=umls_connections ../.venv/bin/python src/main_graph.py
../.venv/bin/python -m knowledge_graph.visualize_entities
```

The `umls_connections` phase is read-only by default:
`materialization_mode=none` and `write_neo4j=false`. Use
`materialization_mode=safe_only` during discovery when you want reports to show
the strict write-eligible subset without changing Neo4j. Set
`KG_UMLS_CONNECTIONS_WRITE_NEO4J=true` only with
`KG_UMLS_CONNECTIONS_MATERIALIZATION_MODE=safe_only` when you want to materialize
strict collapsed UMLS/SNOMED candidate relationships in Neo4j.

### UMLS/SNOMED relation profiles

`knowledge_graph.umls_connections` keeps the core SNOMED relation profile
separate from the first extension so each run is reviewable:

- `core`: `isa`, `inverse_isa`, `has_finding_site`, `finding_site_of`,
  `has_associated_morphology`, `associated_morphology_of`,
  `has_procedure_site`, `has_direct_procedure_site`.
- `first_extension`: `has_definitional_manifestation`,
  `definitional_manifestation_of`, `uses_device`, `device_used_by`,
  `has_direct_device`, `direct_device_of`, `has_measured_component`,
  `measured_component_of`.
- `expanded`: the union of `core` and `first_extension`.
- `balanced_core`: `core` plus `procedure_site_of` and
  `direct_procedure_site_of`.
- `audit_all`: all catalogued audit relations; export-only and never valid with
  `write_neo4j=true`.

Safe read-only commands:

```bash
python3 -m knowledge_graph.umls_connections --doc-id DOC_ID --relation-profile core
python3 -m knowledge_graph.umls_connections --doc-id DOC_ID --relation-profile first_extension
python3 -m knowledge_graph.umls_connections --doc-id DOC_ID --relation-profile expanded
python3 -m knowledge_graph.umls_connections --doc-id DOC_ID --relation-profile core --materialization-mode safe_only --run-name core_safe_only
```

`--include-relation-name` adds relation names to the selected profile and
`--exclude-relation-name` removes them. `--strong-relations-only` remains for
backward compatibility and is equivalent to `--relation-profile expanded`.
`--include-cui` may be repeated to run a targeted CUI subset before `--skip-cui`
and `--max-cuis` are applied. `--replace-existing-connections` is valid only for
a strict `safe_only` write with a selected `doc_id`.

First-extension collapsed candidates are audited with local canonical-type
rules. Compatible forward relations use traversal policy `safe`; compatible
inverse relations use `reverse_review`; incompatible first-extension candidates
use `type_review`. Type-incompatible candidates remain reviewable in the
collapsed JSON, Markdown summary, and materialization report, but are not
materialized.

Local type rules for the first extension:

| relation_name | source representative types | target representative types |
| --- | --- | --- |
| `has_definitional_manifestation` | `disease`, `complication_or_comorbidity` | `clinical_finding` |
| `definitional_manifestation_of` | `clinical_finding` | `disease`, `complication_or_comorbidity` |
| `uses_device` | `procedure_or_intervention`, `diagnostic_test`, `imaging_modality` | `device` |
| `device_used_by` | `device` | `procedure_or_intervention`, `diagnostic_test`, `imaging_modality` |
| `has_direct_device` | `procedure_or_intervention`, `diagnostic_test` | `device` |
| `direct_device_of` | `device` | `procedure_or_intervention`, `diagnostic_test` |
| `has_measured_component` | `diagnostic_test` | `biomarker` |
| `measured_component_of` | `biomarker` | `diagnostic_test` |

For Neo4j Aura, set `NEO4J_URI=neo4j+s://...`, `NEO4J_USERNAME`, and
`NEO4J_PASSWORD` in `.env`. For a local Neo4j server, set
`NEO4J_URI=bolt://localhost:7687`. The standalone KG utilities load `.env`
automatically.

The `scripts/` folder is for cluster execution. Those jobs run through
Slurm and default to `KG_NEO4J_MODE=local`, which starts a local Neo4j
Singularity container on the compute node. This is the expected mode on the
cluster because compute nodes do not have internet access. Use
`KG_NEO4J_MODE=external` only when the cluster/network environment is explicitly
allowed to reach an external Neo4j instance such as Aura.
