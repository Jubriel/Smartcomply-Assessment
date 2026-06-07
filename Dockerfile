FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build tools for wheels (kept minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY main.py ./

# Upgrade packaging tools and install the package (uses pyproject.toml)
RUN pip install --upgrade pip setuptools build \
    && pip install .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--reload"]
