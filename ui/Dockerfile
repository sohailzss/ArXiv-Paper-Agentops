# Gradio UI. Talks to the API over the Docker network only (RAG_API_URL), so it needs no secrets.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    GRADIO_HOST=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    RAG_API_URL=http://api:8000

# Same versions as uv.lock's "ui" group.
RUN pip install --no-cache-dir "gradio==6.31.0" "httpx>=0.28"

WORKDIR /app
COPY ui/gradio_app.py .

EXPOSE 7860
CMD ["python", "gradio_app.py"]
