Note sull'uso:
Commando per lanciare gli script dalla cartella data-etl. Nota che si deve spcificare la fase della pipeline tra : "preprocess", "graph", "entities", "embedings" e "full".
sbatch --export=ALL,KG_PIPELINE_PHASE=embeddings scripts/main_kg.slurm

Ci sono anche dei piccoli script per controllare ciò che è stato caricato sul grafo:
- query_graph, pensato principalmente per investigare i chunk inseriti
- visualize_entitties, pensato per controllare la tipologia di entità caricate.

Per farli eseguire si possono sbatchare i rispettivi file .slurm nella cartella scripts: query_kg.slurm e check_entities.slurm.

I logs di tuta la pipeline etl del knowledge graph si trovano in logs_kg e sono differenziati in base al tipo di script lanciato.