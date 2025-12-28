import json
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RAW_LOGS = ROOT / "data" / "raw_logs" / "logs.jsonl"
OUT = ROOT / "data" / "processed" / "chunks.jsonl"

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def to_chunk(log: dict) -> dict:
    log_id = log.get("id") or f"log_{utc_now_iso()}"
    ts = log.get("timestamp")
    source = log.get("source", "unknown")
    tool = log.get("tool", "unknown")
    tags = log.get("tags", [])
    expected_fix = log.get("expected_fix", "")

    text = clean_text(log.get("text", ""))

    chunk_text = (
        f"[ID] {log_id}\n"
        f"[SOURCE] {source}\n"
        f"[TOOL] {tool}\n"
        f"[TAGS] {', '.join(tags) if tags else ''}\n"
        f"[TIMESTAMP] {ts}\n"
        f"[ERROR_LOG]\n{text}\n"
    ).strip()

    return {
        "chunk_id": f"{log_id}__chunk0",
        "log_id": log_id,
        "created_at": utc_now_iso(),
        "doc_type": "log",
        "source": source,
        "tool": tool,
        "tags": tags,
        "timestamp": ts,
        "text": chunk_text,
        "expected_fix": expected_fix,
        "provenance": log.get("provenance", None),
    }

def main():
    if not RAW_LOGS.exists():
        raise FileNotFoundError(f"Missing {RAW_LOGS}")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0

    with RAW_LOGS.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            log = json.loads(line)

            chunk = to_chunk(log)
            if chunk["text"].strip():
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"Prepared chunks: {n_out} from logs: {n_in}")
    print(f"Wrote: {OUT}")

if __name__ == "__main__":
    main()
