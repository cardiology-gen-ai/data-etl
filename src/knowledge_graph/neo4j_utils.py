"""
neo4j_utils.py

Shared Neo4j utilities for the knowledge graph pipeline.

Responsibilities:
- load Neo4j connection settings from environment variables
- create and verify a Neo4j driver
- safely close the driver
"""

import logging
import os
from typing import Optional

from neo4j import Driver, GraphDatabase


logger = logging.getLogger(__name__)


def _get_optional_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def load_neo4j_config() -> dict:
    """
    Load Neo4j connection settings from environment variables.

    Supported modes:
    - authenticated connection:
        NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    - no-auth local connection:
        NEO4J_URI only
    """
    uri = _get_optional_env("NEO4J_URI", "bolt://localhost:7687")
    username = _get_optional_env("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if uri is None:
        raise RuntimeError("Missing Neo4j environment variable: NEO4J_URI")

    # Enable auth only when both username and password are present.
    auth = (username, password) if username and password else None

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