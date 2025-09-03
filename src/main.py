import os

from huggingface_hub import login, logout

from src.etl_processor import ETLProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "False"

if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(os.getenv("HF_TOKEN"))
    app_id = "cardiology_protocols"
    etl_processor = ETLProcessor(app_id=app_id)

    # etl_processor.perform_etl()

    index_manager = etl_processor.index_manager
    # index_manager.delete_index()
    n_stored_chunks = index_manager.get_n_documents_in_vectorstore()

    try:
        logout()
    except OSError:
        pass