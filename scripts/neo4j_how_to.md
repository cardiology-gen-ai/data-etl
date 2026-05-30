Note sull'uso degli script cluster.

La cartella `scripts/` contiene launcher pensati per Leonardo/Slurm. Altrimenti si può usare direttamente `python src/main_graph.py` come descritto
nel README principale.

Comando per lanciare la pipeline KG dalla cartella `data-etl`:

```bash
sbatch --export=ALL,KG_PIPELINE_PHASE=embeddings scripts/main_kg.slurm
```

Le fasi supportate sono `preprocess`, `graph`, `entities`, `embeddings`,
`normalization`, e `full`.

Su Leonardo il default e `KG_NEO4J_MODE=local`: lo script avvia un container
Neo4j locale via Singularity e forza la connessione runtime a
`LOCAL_NEO4J_URI`, di default `bolt://localhost:7687`. Questo e il modo
consigliato per il cluster, perche i compute node non hanno accesso internet.

Solo se il cluster puo raggiungere un database esterno, ad esempio Neo4j Aura,
si puo usare:

```bash
sbatch --export=ALL,KG_PIPELINE_PHASE=entities,KG_NEO4J_MODE=external scripts/main_kg.slurm
```

In quel caso `.env.leonardo` deve contenere `NEO4J_URI`, `NEO4J_USERNAME`, e
`NEO4J_PASSWORD`, e lo script non avvia il container locale.

Ci sono anche piccoli script per controllare cio che e stato caricato sul grafo:
- `query_kg.slurm`, pensato principalmente per investigare i chunk inseriti.
- `check_entities.slurm`, pensato per controllare la tipologia di entita caricate.

I log della pipeline knowledge graph si trovano in `logs_kg` e sono
differenziati in base al tipo di script lanciato.
