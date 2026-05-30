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
../.venv/bin/python -m knowledge_graph.visualize_entities
```

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


