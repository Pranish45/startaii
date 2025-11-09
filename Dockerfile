# Use python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build tools and rust for compiling pydantic-core when needed
RUN apt-get update && apt-get install -y     build-essential     curl     libssl-dev     pkg-config     && curl https://sh.rustup.rs -sSf | sh -s -- -y     && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_updated.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements_updated.txt

# Copy project files
COPY . .

# Run ingest before starting server
CMD python ingest.py && uvicorn app:app --host 0.0.0.0 --port 8000
