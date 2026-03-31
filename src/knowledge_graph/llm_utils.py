import os
import logging

from dotenv import load_dotenv
from openai import AzureOpenAI


logger = logging.getLogger(__name__)

load_dotenv()


def get_azure_openai_client() -> AzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    print(f"API Key: {'set' if api_key else 'not set'}")
    print(f"Endpoint: {endpoint}")
    print(f"API Version: {api_version}")

    if not all([api_key, endpoint, api_version]):
        raise RuntimeError(
            "Missing Azure OpenAI environment variables: "
            "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION"
        )

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def get_chat_deployment() -> str:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment:
        raise RuntimeError("Missing AZURE_OPENAI_DEPLOYMENT_NAME")
    return deployment


def get_embedding_deployment() -> str:
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    if not deployment:
        raise RuntimeError("Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    return deployment