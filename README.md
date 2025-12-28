RAG Debug Copilot 

RAG Debug Copilot is a lightweight Retrieval-Augmented Generation (RAG) service designed to analyze, classify, and troubleshoot real-world ML / MLOps errors such as Python dependency issues, Vertex AI access errors, CUDA failures, and Airflow runtime problems.

The system combines semantic search (FAISS + sentence embeddings) with LLM-based reasoning (Gemini via Vertex AI), and is fully deployed as a serverless FastAPI service on Google Cloud Run, with structured logging and observability built in.

What does it do?

Given an error message or log snippet, the service:

Retrieves relevant historical error cases using semantic search

Classifies the issue type (e.g. Vertex AI permissions, pip wheel build failure, module not found, GPU/CUDA issues, Airflow runtime errors)

Generates a structured troubleshooting report, including:

Summary of the problem

Likely root causes

Suggested fixes

Diagnostic commands to run next

Falls back gracefully to a deterministic rule-based report if the LLM fails

Architecture (High level)

FastAPI – API layer

FAISS – Vector similarity search

Sentence Transformers – Embedding model (all-MiniLM-L6-v2)

Vertex AI (Gemini) – LLM reasoning layer

Google Cloud Run – Serverless deployment

Cloud Logging & Metrics – Latency, fallback rate, error tracking

API Endpoints:
Health check
GET /health

Semantic search
POST /search
Content-Type: application/json

{
  "query": "ModuleNotFoundError: No module named 'faiss'"
}

Full troubleshooting report
POST /ask
Content-Type: application/json

{
  "query": "Vertex AI: model not found for gemini publisher model"
}

Observability & Metrics

The service emits structured JSON logs to Cloud Logging, including:

Total request latency

Embedding latency

Retrieval latency

LLM latency

Fallback usage (LLM → rule-based)

Error details (if any)

Custom metrics were created to track:

Latency (p50 / p95)

Fallback rate

Error frequency

This makes the service suitable for real-world monitoring and production debugging.


Deployment

The service is containerized and deployed to Google Cloud Run.

Key characteristics:

Auto-scaling (scale-to-zero when idle)

Memory-aware startup (FAISS + embedding model loaded once per instance)

Pay-per-use pricing (very low idle cost)

Notes on Gemini / Vertex AI

Gemini calls are executed via Vertex AI

If access is missing or a model is unavailable in the selected region, the system automatically falls back to a rule-based report

This behavior is intentional and logged explicitly

Local Development
pip install -r requirements.txt
uvicorn app.main:app --reload


Ensure FAISS index files exist:

data/processed/faiss.index
data/processed/faiss_meta.jsonl

Project Structure
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

Why this project?

Production-grade deployment on GCP with an intentionally focused approach on debugging & MLOps scenarios.

Future improvements

Async request handling

Batched embedding inference

Smarter fallback triggering

Expand the retrieval corpus with large-scale, high-quality debugging sources (e.g. Stack Overflow–style discussions, GitHub issues, CI/CD failure logs)

Frontend UI for interactive debugging

GitHub Actions (linting, tests, CI/CD)
