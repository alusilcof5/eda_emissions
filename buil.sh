#!/usr/bin/env bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

# Install dependencies using Poetry
poetry install

# OPTIONAL: Run migrations or prepare anything needed
# poetry run python manage.py migrate  # para Django, por ejemplo
