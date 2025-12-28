# RAG Debug Copilot

RAG Debug Copilot is a lightweight **Retrieval-Augmented Generation (RAG)** service designed to analyze, classify, and troubleshoot real-world **ML / MLOps errors**, including Python dependency issues, Vertex AI access errors, CUDA failures, and Airflow runtime problems.

The system combines **semantic retrieval** (FAISS + sentence embeddings) with **LLM-based reasoning** (Gemini via Vertex AI) and is deployed as a **serverless FastAPI service on Google Cloud Run**, with structured logging and observability built in.

---

## What Does It Do?

Given an error message or log snippet, the service:

- Retrieves relevant historical error cases using semantic search
- Classifies the issue type (e.g. Vertex AI permissions, pip wheel build failures, missing Python modules, GPU/CUDA issues, Airflow runtime errors)
- Generates a structured troubleshooting report containing:
  - A concise summary of the problem
  - Likely root causes
  - Suggested fixes
  - Diagnostic commands to run next
- Falls back gracefully to a deterministic, rule-based report if the LLM fails or is unavailable

---

## High-Level Architecture

- **FastAPI** – API layer
- **FAISS** – Vector similarity search
- **Sentence Transformers** – Embedding model (`all-MiniLM-L6-v2`)
- **Vertex AI (Gemini)** – LLM reasoning layer
- **Google Cloud Run** – Serverless deployment
- **Cloud Logging & Metrics** – Latency, fallback rate, error tracking

---

## API Endpoints

### Health Check
GET /health

POST /search
Content-Type: application/json

### Semantic Search
POST /search
Content-Type: application/json

{
  "query": "ModuleNotFoundError: No module named 'faiss'"
}

### Full Troubleshooting Report

POST /ask
Content-Type: application/json

{
  "query": "Vertex AI: model not found for gemini publisher model"
}



## Observability & Metrics

The service emits **structured JSON logs** to Google Cloud Logging for every request.

Each request logs:

- Total request latency
- Embedding latency
- Retrieval latency
- LLM latency (when applicable)
- Fallback usage (LLM → rule-based)
- Error details (if any)

These logs are used to create **log-based metrics** for monitoring and analysis.

### Custom Metrics

The following custom metrics were created from logs:

- **Latency** (distribution metric)
  - p50 (median latency)
  - p95 (tail latency)
- **Fallback rate**
  - Percentage of requests handled by the rule-based fallback
- **Error frequency**
  - Count of failed or partially failed requests

This setup enables production-style observability and debugging using Google Cloud’s native monitoring tools.

---

## Deployment

The service is containerized and deployed to **Google Cloud Run**.

Key characteristics:

- Auto-scaling with scale-to-zero when idle
- Memory-aware startup (FAISS index and embedding model loaded once per instance)
- Pay-per-use pricing with minimal idle cost

---

## Notes on Gemini / Vertex AI

- Gemini calls are executed via **Vertex AI**
- If access is missing or a model is unavailable in the selected region, the system automatically falls back to a rule-based report
- This behavior is intentional and explicitly logged for observability

---

## Local Development

Install dependencies:
 - pip install -r requirements.txt

Run the API locally:
 - uvicorn app.main:app --reload

Ensure FAISS index files exist:
 - data/processed/faiss.index
 - data/processed/faiss_meta.jsonl


## Project Structure

rag_workflow/
├── app/
│   └── main.py
├── data/
│   └── processed/
├── scripts/
│   ├── build_faiss_index.py
│   └── prepare_logs.py
├── Dockerfile
├── requirements.txt
└── README.md

## Why This Project?

This project demonstrates a **production-oriented RAG system** deployed on GCP, with a deliberate focus on **debugging and MLOps failure scenarios**.

It emphasizes:

- Grounded retrieval over hallucination
- Explicit fallback behavior
- Observability-first design
- Realistic deployment constraints (memory, cold starts, costs)

## Future Improvements

- Async request handling
- Batched embedding inference
- Smarter fallback triggering
- Expand the retrieval corpus with large-scale, high-quality debugging sources (e.g. Stack Overflow–style discussions, GitHub issues, CI/CD failure logs)
- Frontend UI for interactive debugging
- CI/CD and quality checks (linting, tests, GitHub Actions)



