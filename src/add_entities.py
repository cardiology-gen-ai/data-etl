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
from typing import List, Dict, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import AzureOpenAI


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
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
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def extract_concepts(text: str) -> List[Dict[str, str]]:
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    try:
        data = parse_llm_json(content)
        if not isinstance(data, list):
            raise ValueError

        return [
            {"name": d["name"].strip(), "type": d["type"].strip()}
            for d in data
            if isinstance(d, dict) and "name" in d and "type" in d
        ]

    except Exception:
        logger.error("Failed to parse LLM output: %s", content)
        return []

# Neo4j helpers
def enforce_constraint(tx):
    tx.run(
        """
        CREATE CONSTRAINT concept_name IF NOT EXISTS
        FOR (c:Concept)
        REQUIRE c.name IS UNIQUE
        """
    )

def create_concept_and_link(tx, section_uid: str, concept: Dict[str, str]):
    tx.run(
        """
        MATCH (s:Section {uid: $uid})
        MERGE (c:Concept {name: $name})
        SET c.type = $type
        MERGE (s)-[:MENTIONS]->(c)
        """,
        uid=section_uid,
        name=concept["name"],
        type=concept["type"],
    )

def add_entities_from_sections(
    doc_id: Optional[str] = None,
    use_section_text: bool = False,
    max_sections: Optional[int] = None,
):
    """
    Extract concepts from section titles (and optionally text)
    and attach them to the Neo4j graph.

    If doc_id is provided, only that document is processed.
    """

    with driver.session() as session:

        query = """
        MATCH (s:Section)
        WHERE $doc_id IS NULL OR s.doc_id = $doc_id
        RETURN s.uid AS uid,
               s.doc_id AS doc_id,
               s.section_id AS section_id,
               s.title AS title,
               s.text AS text
        ORDER BY s.uid
        """

        rows = list(session.run(query, doc_id=doc_id))

        if max_sections:
            rows = rows[:max_sections]

        logger.info(
            "Processing %d sections%s",
            len(rows),
            f" for document {doc_id}" if doc_id else "",
        )

        for row in rows:
            section_uid = row["uid"]
            source_text = row["title"] or ""

            if use_section_text and row["text"]:
                source_text += "\n" + row["text"]

            if not source_text.strip():
                continue

            logger.info(
                "Extracting entities | doc=%s section=%s",
                row["doc_id"],
                row["section_id"],
            )

            concepts = extract_concepts(source_text)

            if not concepts:
                continue

            logger.info(
                "  → %d concepts: %s",
                len(concepts),
                ", ".join(c["name"] for c in concepts),
            )

            for concept in concepts:
                session.execute_write(
                    create_concept_and_link,
                    section_uid,
                    concept,
                )

    logger.info("Entity extraction completed")



if __name__ == "__main__":
    
    with driver.session() as session:
        session.execute_write(enforce_constraint)

    add_entities_from_sections(
        doc_id=None,            # set to specific doc_id if needed
        use_section_text=False, # start with titles only
        max_sections=None,
    )
