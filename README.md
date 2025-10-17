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
