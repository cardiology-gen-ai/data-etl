"""
add_entities.py

Adds Concept nodes to an existing Neo4j graph by extracting
medical concepts from section titles and/or section text
using Azure OpenAI.

Nodes:
  (:Concept {name, type})

Relationships:
  (:Section)-[:MENTIONS]->(:Concept)
"""

import os
import json
import re
import logging
from typing import List, Dict

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import AzureOpenAI


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("entity_extraction")


load_dotenv()

# Azure OpenAI
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

if not all([AZURE_DEPLOYMENT, AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION]):
    raise RuntimeError("Missing Azure OpenAI environment variables")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise RuntimeError("Missing Neo4j environment variables")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)



# LLM prompt for concept (entity) extraction

SYSTEM_PROMPT = """You are a medical terminology expert.

Extract normalized, reusable cardiology-related concepts.

Rules:
- Use lowercase
- Use singular form
- Prefer standard clinical terminology
- Do NOT include trivial words (e.g. management, approach, recommendations)
- Merge synonyms (e.g. HCM -> hypertrophic cardiomyopathy)

For each concept, assign ONE type from:
- disease
- phenotype
- diagnostic_test
- imaging_modality
- management
- risk_factor

Return JSON only.
"""

USER_PROMPT_TEMPLATE = """
Text:
\"\"\"
{text}
\"\"\"

Return a JSON array of objects.
Example:
[
  {{"name": "hypertrophic cardiomyopathy", "type": "disease"}},
  {{"name": "sudden cardiac death", "type": "risk_factor"}}
]
"""

# Helpers

def parse_llm_json(text: str):
    """
    Extract and parse JSON from LLM output that may be wrapped in ```json fences.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    return json.loads(text)


def extract_concepts(text: str) -> List[Dict[str, str]]:
    """
    Call Azure OpenAI and extract typed medical concepts.
    """
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(text=text),
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    try:
        data = parse_llm_json(content)
        if not isinstance(data, list):
            raise ValueError

        concepts: List[Dict[str, str]] = []
        for item in data:
            if (
                isinstance(item, dict)
                and "name" in item
                and "type" in item
                and isinstance(item["name"], str)
                and isinstance(item["type"], str)
            ):
                concepts.append(
                    {
                        "name": item["name"].strip(),
                        "type": item["type"].strip(),
                    }
                )

        return concepts

    except Exception:
        logger.error("Failed to parse LLM output: %s", content)
        return []


# Neo4j helpers
def create_concept_and_link(tx, section_id: str, concept: Dict[str, str]):
    tx.run(
        """
        MATCH (s:Section {section_id: $section_id})
        MERGE (c:Concept {name: $name})
        SET c.type = $type
        MERGE (s)-[:MENTIONS]->(c)
        """,
        section_id=section_id,
        name=concept["name"],
        type=concept["type"],
    )


# Add entities from section titles #TODO: later on consider doing so on section content as well
def add_entities_from_sections(
    use_section_text: bool = False,
    max_sections: int | None = None,
):
    """
    Extract concepts from section titles (and optionally section text)
    and attach them to the Neo4j graph.
    """

    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Section)
            RETURN s.section_id AS id,
                   s.title AS title,
                   s.text AS text
            ORDER BY s.section_id
            """
        )

        rows = list(result)
        if max_sections:
            rows = rows[:max_sections]

        logger.info("Processing %d sections", len(rows))

        for row in rows:
            section_id = row["id"]

            source_text = row["title"] or ""
            if use_section_text and row["text"]:
                source_text += "\n" + row["text"]

            if not source_text.strip():
                continue

            logger.info("Extracting entities from section %s", section_id)

            concepts = extract_concepts(source_text)

            if not concepts:
                continue

            logger.info(
                "  → found %d concepts: %s",
                len(concepts),
                ", ".join(c["name"] for c in concepts),
            )

            for concept in concepts:
                session.execute_write(
                    create_concept_and_link,
                    section_id,
                    concept,
                )

    logger.info("Entity extraction completed")



if __name__ == "__main__":
    add_entities_from_sections(
        use_section_text=False,  # Starts with section titles only
        max_sections=None,       
    )
