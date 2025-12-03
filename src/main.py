import os

import huggingface_hub

from src.etl_processor import ETLProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "False"


if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(os.getenv("HF_TOKEN"))
    app_id = "cardiology_protocols"
    etl_processor = ETLProcessor(app_id=app_id)

    etl_processor.perform_etl(force_md_conv=True) # Use existing Markdown files if available

    try:
        huggingface_hub.logout()
    except OSError:
        pass