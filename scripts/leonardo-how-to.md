## Running this app in Leonardo Cluster (by Cineca)

### Leonardo env file 
The `.env` file read in all the following scripts is assumed to be called `.env.leonardo` , which populates the fields described in `.env.template`, namely:
```
CONFIG_PATH=  # e.g. /leonardo_work/IscrC_CardioAI/data-etl/config.json
APP_CONFIG_PATH=  # e.g. /leonardo_work/IscrC_CardioAI/cardiology-gen-ai/config.json
HF_TOKEN= # your hugging face token 
QDRANT_URL=  # e.g. http://localhost:6333
INDEX_ROOT=  # e.g. /leonardo_work/IscrC_CardioAI/cardiology-gen-ai/index
KG_NEO4J_MODE=local
LOCAL_NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=
```

`KG_NEO4J_MODE=local` is the expected cluster default. It starts the Neo4j
Singularity container on the compute node and avoids relying on internet access.
Use `KG_NEO4J_MODE=external` only if the cluster/network can reach an external
Neo4j database and `.env.leonardo` contains the external `NEO4J_URI` and
credentials.

1. Install all project dependencies using the script `install.sh`. From the login node, proceed e.g. as:
    ```bash
    cd $WORK/data-etl
    chmod u+x scripts/install.sh
    scripts/install.sh
    ```

2. Download all HuggingFace models in the login node, as the compute node does not have internet connection. 
    Do it using the script `hf_init.sh`, from the login node proceed e.g. as:
    ```bash
    cd $WORK/data-etl
    chmod u+x scripts/hf_init.sh
    scripts/hf_init.sh
    ```
   
   Optional UMLS/scispaCy resources also need to be prepared before a cluster
   normalization run, because the first linker run downloads a large local cache.
   If the cache is not already available on the cluster filesystem, run
   normalization on a machine with internet or copy the prepared scispaCy cache
   into the cluster user cache before submitting the Slurm job.

3. Launch the python script of interest (e.g. `src/main.py`), possibly activating qdrant using singularity, using the script `main.slurm`.
    From the user login page proceed e.g. as:
    ```bash
    cd $WORK/data-etl
    sbatch scripts/main.slurm
    ```
   If you need to use qdrant, make sure to perform (once, the first time you run the code):
    ```bash
   cd containers # or mkdir containers 
   singularity pull qdrant.sif docker://qdrant/qdrant:latest 
    ```
   from your working directory (e.g. `$WORK` in this case). 
   
### NB

At the moment, all the scripts assume that the `$WORK` directory of the CardioAI IscraC project reflects the structure of the cardiology-gen-ai GitHub organization.
Especially, make sure that all the `config.json` are properly set.
