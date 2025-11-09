FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential curl libssl-dev pkg-config
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . .

CMD python ingest.py && uvicorn app:app --host 0.0.0.0 --port 8000
