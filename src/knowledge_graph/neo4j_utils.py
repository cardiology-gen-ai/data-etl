"""
neo4j_utils.py

Shared Neo4j utilities for the knowledge graph pipeline.

Responsibilities:
- load Neo4j connection settings from environment variables
- create and verify a Neo4j driver
- safely close the driver
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver


logger = logging.getLogger(__name__)


def load_neo4j_config() -> dict:
    """
    Load Neo4j connection settings from environment variables.

    Supported modes:
    - authenticated connection:
        NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    - no-auth local connection:
        NEO4J_URI, optional NEO4J_USERNAME, empty/missing NEO4J_PASSWORD
    """
    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    username = os.getenv("NEO4J_USERNAME", "neo4j").strip()
    password = os.getenv("NEO4J_PASSWORD", "")

    if not uri:
        raise RuntimeError("Missing Neo4j environment variable: NEO4J_URI")

    # For local single-instance Neo4j, prefer direct bolt connection
    if uri.startswith("neo4j://"):
        uri = uri.replace("neo4j://", "bolt://", 1)
        logger.info("Switched Neo4j URI to direct bolt connection: %s", uri)

    auth = (username, password) if password else None

    return {
        "uri": uri,
        "username": username,
        "password": password,
        "auth": auth,
    }


def get_neo4j_driver(verify: bool = True) -> Driver:
    """
    Create and optionally verify a Neo4j driver.
    """
    config = load_neo4j_config()

    logger.info(
        "Connecting to Neo4j | uri=%s | auth=%s",
        config["uri"],
        "enabled" if config["auth"] is not None else "disabled",
    )

    driver = GraphDatabase.driver(
        config["uri"],
        auth=config["auth"],
    )

    if verify:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j")

    return driver


def close_driver(driver: Optional[Driver]) -> None:
    """
    Safely close a Neo4j driver if it exists.
    """
    if driver is None:
        return

    try:
        driver.close()
        logger.info("Neo4j driver closed")
    except Exception as e:
        logger.warning("Failed to close Neo4j driver cleanly: %s", e)