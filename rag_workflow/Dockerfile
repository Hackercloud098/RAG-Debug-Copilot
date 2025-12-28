FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + data (includes FAISS index files)
COPY . .

# Cloud Run uses PORT env var, default 8080
ENV PORT=8080
EXPOSE 8080


#CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
#CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]


#CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
