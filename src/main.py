import os

import huggingface_hub

from src.etl_processor import ETLProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["OLLAMA_HOST"] = "http://ollama:11434"


if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(os.getenv("HF_TOKEN"))
    app_id = "cardiology_protocols"
    etl_processor = ETLProcessor(app_id=app_id)
    print(etl_processor.index_manager.get_n_documents_in_vectorstore())
    # metadata_path = etl_processor.markdown_converter.config.output_folder.folder / "documents_metadata_qwen3-embedding:8b.json"
    # etl_processor.perform_etl(force_md_conv=False, existing_metadata_path=metadata_path)

    try:
        huggingface_hub.logout()
    except OSError:
        pass