"""
neo4j_utils.py

Shared Neo4j utilities for the knowledge graph pipeline.

Responsibilities:
- load Neo4j credentials from environment variables
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
    """
    load_dotenv()

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        raise RuntimeError(
            "Missing Neo4j environment variables: "
            "NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
        )

    return {
        "uri": uri,
        "username": username,
        "password": password,
    }


def get_neo4j_driver(verify: bool = True) -> Driver:
    """
    Create and optionally verify a Neo4j driver.
    """
    config = load_neo4j_config()

    driver = GraphDatabase.driver(
        config["uri"],
        auth=(config["username"], config["password"]),
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