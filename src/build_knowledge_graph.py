import os
import pathlib
from neo4j import GraphDatabase

from knowledge_graph.build_kg import build_kg, KGPaths
from knowledge_graph.recommendation_extraction_manager import RecommendationExtractionManager

EXTRACT_RECOMMENDATIONS = False

app_id = "cardiology_protocols"
storage = {
    "base": "data/cvd",
    "original_files": "pdfdocs",
    "preprocessing_output": "mddocs",
    "kg_folder": "kgdocs",
}
base_folder = pathlib.Path(pathlib.Path.cwd() / storage["base"])
preprocessing_output_folder = base_folder / storage["preprocessing_output"]
kg_output_folder = base_folder / storage["kg_folder"]
header_levels = 1

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
)
driver.verify_connectivity()

paths = KGPaths(
    preprocessing_output=preprocessing_output_folder,
    kg_folder=kg_output_folder,
    chunks_folder=preprocessing_output_folder / "tocs" / f"chunks_{header_levels}",
)

input_files = [pathlib.Path(f) for f in os.listdir((base_folder / storage["original_files"]).as_posix()) if f.lower().endswith("pdf")]
doc_ids = [f.stem for f in input_files]

llm_deployment = "gpt-4o"
embedding_deployment = "text-embedding-3-small"

if EXTRACT_RECOMMENDATIONS:
    rec_extractor = RecommendationExtractionManager(
        output_folder=kg_output_folder/ "recommendations",
        tabs_folder=preprocessing_output_folder,
        model=llm_deployment,
        app_id=app_id
    )
    for file in input_files:
        rec_catalog = rec_extractor(pathlib.Path(file))
        print(rec_catalog.stats())

SKIP_PROSE_ENTITIES = False
SKIP_RECOMMENDATIONS = False
SKIP_SECTION_LINKS = False

try:
    summary = build_kg(
        driver=driver,
        paths=paths,
        doc_ids=doc_ids,
        # add_entities_kwargs={"model_name": llm_deployment},
        # UMLS
        umls_mode="api",  # "hybrid",
        umls_scispacy_kwargs={"model_name": "en_core_sci_scibert", "threshold": 0.85},
        umls_api_kwargs={"threshold": 0.80},
        # Embeddings
        embedding_provider="openai",
        embedding_model=embedding_deployment,
        embedding_dimensions=1536,

        replace_existing_document = True,
        skip_prose_entities = SKIP_PROSE_ENTITIES,
        skip_recommendations = SKIP_RECOMMENDATIONS,
        skip_section_links = SKIP_SECTION_LINKS,

        skip_disambiguation = False,
        skip_umls_normalization = False,
        skip_embeddings = False,
        skip_sanity_checks = False,
    )
finally:
    driver.close()