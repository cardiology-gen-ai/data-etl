ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/

RUN pip install .

COPY src/ /app/src
# COPY config.json /app/config.json  # TODO: uncomment in production

# TODO: modify this in production
USER root
