"""
Document Splitter.
Splits a .md document in chunks using LangChain.
"""
import itertools
import os

from src.utils.logger import get_logger
from src.config.manager import ETLConfigManager
from src.managers.chunking_manager import ChunkingManager

def main():
    logger = get_logger("Splitter for markdown documents")
    app_id = "cardiology_protocols"
    config = ETLConfigManager(app_id=app_id).config.preprocessing
    md_files_folder = config.output_folder.folder
    logger.info(f"Directory containing markdown files: {md_files_folder}")
    md_files = list(itertools.chain.from_iterable(
        [[f for f in os.listdir(config.output_folder.folder.as_posix()) if f.lower().endswith(allowed_extension)]
         for allowed_extension in config.output_folder.allowed_extensions]))
    splitter_config = config.chunking_manager
    splitter = ChunkingManager(splitter_list=splitter_config.splitter)
    for file in md_files:
        chunks = splitter(md_files_folder / file)

if __name__ == "__main__":
    main()
