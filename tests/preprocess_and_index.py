import os
import pathlib

import huggingface_hub

from etl_processor import ETLProcessor

CONFIG_FOLDER = pathlib.Path("/Users/giai/Desktop/repos/cardiology-gen-ai/data-etl/tests/configs/all")
CONFIG_FILE_NAME = "flat_bm25.json"

if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(os.getenv("HF_TOKEN"))
    app_id = "cardiology_protocols"
    # config_files = [f for f in os.listdir(CONFIG_FOLDER.as_posix()) if f.lower().endswith("json")]
    config_path = CONFIG_FOLDER / CONFIG_FILE_NAME
    etl_processor = ETLProcessor(app_id=app_id, config_path=config_path.as_posix())
    etl_processor.perform_etl(force_md_conv=True)
    try:
        huggingface_hub.logout()
    except OSError:
        pass