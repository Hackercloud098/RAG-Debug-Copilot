from fastapi import FastAPI
from pydantic import BaseModel
import time
import uuid
from pathlib import Path
import json
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import os
import sys




app = FastAPI(title="RAGDebugCopilot")


class AskRequest(BaseModel):
    query: str
    force_fallback: bool = False



@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


ROOT = Path(__file__).resolve().parents[1]
FAISS_INDEX_PATH = ROOT / "data" / "processed" / "faiss.index"
FAISS_META_PATH = ROOT / "data" / "processed" / "faiss_meta.jsonl"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_faiss_index = None
_faiss_meta: List[Dict] | None = None
_emb_model = None


def load_faiss():
    global _faiss_index, _faiss_meta, _emb_model

    if _faiss_index is None:
        if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
            return None, None, None

        _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

        _faiss_meta = []
        with FAISS_META_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    _faiss_meta.append(json.loads(line))

    if _emb_model is None:
        _emb_model = SentenceTransformer(EMB_MODEL_NAME)

    return _faiss_index, _faiss_meta, _emb_model


def semantic_search(query: str, k: int = 3):
    index, meta, model = load_faiss()
    if index is None:
        return [], {"error": "FAISS index not found. Run: python scripts/build_faiss_index.py"}

    t0 = time.perf_counter()
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    t_embed_ms = round((time.perf_counter() - t0) * 1000, 2)

    t1 = time.perf_counter()
    scores, ids = index.search(q_emb, k)
    t_search_ms = round((time.perf_counter() - t1) * 1000, 2)

    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        m = meta[int(idx)]
        hits.append({
            "score": float(score),
            "chunk_id": m.get("chunk_id"),
            "source": m.get("source"),
            "tool": m.get("tool"),
            "tags": m.get("tags", []),
            "preview": (m.get("text", "")[:400] + ("..." if len(m.get("text", "")) > 400 else "")),
        })

    return hits, {"embed_ms": t_embed_ms, "search_ms": t_search_ms}



def classify_issue(query: str, hits: List[Dict]) -> str:
    """Simple classifier using query + retrieved tags."""
    q = query.lower()
    tags = set()
    for h in hits:
        for t in h.get("tags", []):
            tags.add(str(t).lower())

    # Gemini access
    if "gemini" in q or "publisher-model" in tags or "model-access" in tags or "vertex-ai" in tags:
        return "vertex_model_access"

    # Wheel build failures
    if "wheel" in q or "build" in q or "could not build wheels" in q or "subprocess-exited-with-error" in q or "native-extension" in tags:
        return "pip_wheel_build"

    # Module not found
    if "modulenotfounderror" in q or "no module named" in q or "module-not-found" in tags:
        return "python_module_not_found"

    # CUDA/GPU
    if "cuda" in q or "cudnn" in q or "gpu" in q or "cuBLAS".lower() in q or "cuFFT".lower() in q or "cuda" in tags:
        return "cuda_gpu"

    # Airflow
    if "airflow" in q or "airflow" in tags:
        return "airflow_runtime"

    return "unknown"


def generate_debug_report(query: str, hits: List[Dict]) -> Dict:
    issue = classify_issue(query, hits)

    report = {
        "issue_type": issue,
        "summary": "",
        "likely_root_causes": [],
        "suggested_fixes": [],
        "next_diagnostic_commands": [],
    }

    if issue == "vertex_model_access":
        report["summary"] = "Vertex AI returned a model not found / access error for a Gemini publisher model."
        report["likely_root_causes"] = [
            "Wrong model name or location (region mismatch).",
            "Vertex AI API not enabled or request sent to a region where the model ID isn’t available.",
            "Project/account doesn’t have access to that publisher model (billing/org policy/permissions).",
        ]
        report["suggested_fixes"] = [
            "Confirm the model name and region you’re calling (e.g., us-central1 vs europe-west4).",
            "Verify Vertex AI API is enabled and your project has billing enabled.",
            "Check IAM permissions for Vertex AI usage (and any org policy restrictions).",
            "List available models in that region (or switch to a known-available model ID).",
        ]
        report["next_diagnostic_commands"] = [
            "gcloud config get-value project",
            "gcloud services list --enabled | grep aiplatform",
            "gcloud auth list",
            "gcloud ai locations list",
        ]

    elif issue == "pip_wheel_build":
        report["summary"] = "pip failed to build a wheel from source (common with native extensions)."
        report["likely_root_causes"] = [
            "Missing system build tools / compilers (gcc/clang, build-essential, MSVC on Windows).",
            "Missing Python headers or a dependency required during compilation.",
            "Package doesn’t provide wheels for your Python version / platform, so pip tries to compile.",
        ]
        report["suggested_fixes"] = [
            "Upgrade pip/setuptools/wheel first.",
            "Install platform build tools (Linux: build-essential; macOS: Xcode CLI tools; Windows: Build Tools for Visual Studio).",
            "Try a supported Python version (sometimes older packages don’t build on newest Python).",
            "Look for a prebuilt wheel / alternative package if compilation keeps failing.",
        ]
        report["next_diagnostic_commands"] = [
            "python -V",
            "pip -V",
            "pip install -U pip setuptools wheel",
            "pip install <package> -v",
        ]

    elif issue == "python_module_not_found":
        report["summary"] = "A package appears installed but Python cannot import it (environment/interpreter mismatch)."
        report["likely_root_causes"] = [
            "Installed into a different Python environment than the one running your script.",
            "Multiple Pythons/venvs on PATH; `pip` points to one interpreter and `python` points to another.",
            "Your script name or folder shadows the module name.",
        ]
        report["suggested_fixes"] = [
            "Use `python -m pip install ...` to force pip to match the interpreter.",
            "Check `which python` / `which pip` (or `where` on Windows).",
            "Recreate the venv and reinstall dependencies if things are inconsistent.",
        ]
        report["next_diagnostic_commands"] = [
            "python -c \"import sys; print(sys.executable)\"",
            "python -m pip -V",
            "pip -V",
            "python -c \"import site; print(site.getsitepackages())\"",
        ]

    elif issue == "cuda_gpu":
        report["summary"] = "CUDA / GPU runtime is not configured correctly or conflicting GPU plugins were detected."
        report["likely_root_causes"] = [
            "CUDA/cuDNN versions don’t match the framework build (TensorFlow/PyTorch).",
            "GPU libraries are duplicated/registered multiple times (common in mixed installs).",
            "No compatible GPU driver or CUDA runtime is available on the machine/container.",
        ]
        report["suggested_fixes"] = [
            "Verify GPU driver + CUDA/cuDNN versions match the framework requirements.",
            "Prefer official installation paths (conda or official wheels) for GPU builds.",
            "Avoid mixing system CUDA + pip CUDA packages unless you know the compatibility matrix.",
        ]
        report["next_diagnostic_commands"] = [
            "nvidia-smi",
            "python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"",
            "python -c \"import torch; print(torch.cuda.is_available())\"",
        ]

    elif issue == "airflow_runtime":
        report["summary"] = "Airflow failed to start due to missing dependencies in its runtime environment."
        report["likely_root_causes"] = [
            "Airflow running under an old Python (e.g., Python 2.7) with incompatible dependencies.",
            "Missing `cryptography`/`fernet` dependency required by config or security utilities.",
            "Broken/partial installation inside the Airflow environment.",
        ]
        report["suggested_fixes"] = [
            "Use a supported Airflow + Python version (modern Airflow requires Python 3.x).",
            "Install missing dependency into the same environment running airflow.",
            "Consider using official Airflow Docker images to avoid system Python mismatches.",
        ]
        report["next_diagnostic_commands"] = [
            "airflow version",
            "python -V",
            "pip show cryptography",
            "which airflow && which python && which pip",
        ]

    else:
        report["summary"] = "Could not confidently classify the issue from the current query and evidence."
        report["likely_root_causes"] = [
            "Not enough context in the query/log snippet.",
        ]
        report["suggested_fixes"] = [
            "Paste the full stack trace and the command you ran (plus OS + Python version).",
        ]
        report["next_diagnostic_commands"] = [
            "python -V",
            "pip -V",
        ]

    return report


GEMINI_MODEL = "gemini-2.0-flash-001"

_genai_client = None

def get_genai_client():
    global _genai_client
    if _genai_client is None:
        # env vars: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_VERTEXAI
        _genai_client = genai.Client()
    return _genai_client


def generate_report_with_gemini(query: str, hits: List[Dict]) -> Dict:
    """
    Returns a dict with keys:
    issue_type, summary, likely_root_causes, suggested_fixes, next_diagnostic_commands
    """
    client = get_genai_client()

    evidence_blocks = []
    for i, h in enumerate(hits[:3], start=1):
        evidence_blocks.append(
            f"Evidence {i}:\n"
            f"- chunk_id: {h.get('chunk_id')}\n"
            f"- source: {h.get('source')}\n"
            f"- tool: {h.get('tool')}\n"
            f"- tags: {h.get('tags')}\n"
            f"- score: {h.get('score')}\n"
            f"- preview:\n{h.get('preview')}\n"
        )
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "unknown")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "unknown")

    prompt = f"""

System context:
- project: {project}
- location: {location}

You are RAGDebugCopilot. You must produce a troubleshooting report in STRICT JSON.

User query:
{query}

Retrieved evidence:
{chr(10).join(evidence_blocks)}

Return ONLY valid JSON with EXACTLY these keys:
- issue_type: one of ["vertex_model_access","pip_wheel_build","python_module_not_found","cuda_gpu","airflow_runtime","unknown"]
- summary: string
- likely_root_causes: array of strings (3-6)
- suggested_fixes: array of strings (3-8)
- next_diagnostic_commands: array of strings (3-10)

Rules:
- Base your report on the retrieved evidence; don't invent product-specific commands that aren't reasonable.
- Keep steps actionable and concise.
- NEVER invent project IDs, regions, or model names. If not explicitly present in the user query or evidence, write "unknown".
- Do not mention any project other than what appears in the evidence.


Do not wrap the JSON in markdown backticks. Output raw JSON only.
"""

    t0 = time.perf_counter()
    resp = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json"
        )
    )

    t_llm_ms = round((time.perf_counter() - t0) * 1000, 2)


    text = (resp.text or "").strip()

    try:
        report = json.loads(text)
    except Exception:
        report = {
            "issue_type": "unknown",
            "summary": "Gemini response was not valid JSON; falling back to deterministic report.",
            "likely_root_causes": [],
            "suggested_fixes": [],
            "next_diagnostic_commands": [],
        }

    return report, t_llm_ms


def log_event(payload: Dict):
    """Print one JSON line to stdout (Cloud Run/Cloud Logging friendly)."""
    payload["service"] = "ragdebugcopilot"
    payload["env"] = os.getenv("ENV", "local")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


# Endpoints

@app.post("/search")
def search(req: AskRequest):
    t0 = time.perf_counter()
    hits, timings = semantic_search(req.query, k=3)
    t_total_ms = round((time.perf_counter() - t0) * 1000, 2)

    log_event({
        "event": "search",
        "request_id": str(uuid.uuid4()),
        "latency_ms": t_total_ms,
        "embed_ms": timings.get("embed_ms"),
        "search_ms": timings.get("search_ms"),
        "num_hits": len(hits),
    })

    return {"query": req.query, "num_hits": len(hits), "timings_ms": timings, "hits": hits}




@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())

    hits, rt = semantic_search(req.query, k=3)
    hits = [h for h in hits if h["score"] >= 0.3]

    use_gemini = not req.force_fallback

    t_llm_ms = None

    llm_error = None

    if use_gemini:
        try:
            report, t_llm_ms = generate_report_with_gemini(req.query, hits)
        except Exception as e:
            llm_error = repr(e)
            report = generate_debug_report(req.query, hits)
    else:
        report = generate_debug_report(req.query, hits)

    
    model_used = "gemini" if (use_gemini and t_llm_ms is not None) else "rule_based"


    t_total_ms = round((time.perf_counter() - t0) * 1000, 2)

    timings = {"total": t_total_ms, **rt}
    if t_llm_ms is not None:
        timings["llm_ms"] = t_llm_ms

    top_score = hits[0]["score"] if hits else None
    issue_type = report.get("issue_type") if isinstance(report, dict) else None

    fallback_used = (model_used != "gemini")

    log_event({
        "event": "ask",
        "request_id": request_id,
        "model_used": model_used,
        "fallback_used": fallback_used,
        "latency_ms": t_total_ms,
        "embed_ms": rt.get("embed_ms"),
        "search_ms": rt.get("search_ms"),
        "llm_ms": t_llm_ms,
        "issue_type": issue_type,
        "num_citations": len(hits),
        "top_score": top_score,
        "llm_error": llm_error,
    })



    return {
        "request_id": request_id,
        "model_used": model_used,
        "query": req.query,
        "report": report,
        "timings_ms": timings,
        "llm_error": llm_error,
        "citations": [
            {
                "chunk_id": h["chunk_id"],
                "source": h["source"],
                "tool": h["tool"],
                "tags": h["tags"],
                "score": h["score"],
            }
            for h in hits
        ],
        
    }

@app.on_event("startup")
def warmup():
    load_faiss()
    _, _, model = load_faiss()
    if model:
        model.encode(["warmup"], normalize_embeddings=True)

    #try:
    #    client = get_genai_client()
    #    client.models.generate_content(model=GEMINI_MODEL, contents="Respond with OK")
    #except Exception:
    #    pass