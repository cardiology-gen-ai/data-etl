import os

import huggingface_hub

from src.etl_processor import ETLProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "False"

# TODO: change when dockerizing
# sudo uv pip install -e ../cardiology-gen-ai

if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(os.getenv("HF_TOKEN"))
    app_id = "cardiology_protocols"
    etl_processor = ETLProcessor(app_id=app_id)

    # etl_processor.perform_etl()

    index_manager = etl_processor.index_manager
    # index_manager.delete_index()
    n_stored_chunks = index_manager.get_n_documents_in_vectorstore()

    try:
        huggingface_hub.logout()
    except OSError:
        pass