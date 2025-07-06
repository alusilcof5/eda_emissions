#!/usr/bin/env bash
# Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

# Instalar dependencias del proyecto
poetry install
