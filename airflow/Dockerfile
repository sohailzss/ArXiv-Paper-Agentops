FROM apache/airflow:3.3.2-python3.12

USER root
# libgl1/libglib2.0-0: needed by Docling's image libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# 1) CPU-only PyTorch first, so Docling doesn't pull the ~2 GB CUDA build.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) Project deps, keeping Airflow pinned to the image's version.
COPY airflow/requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements-airflow.txt

# 3) Pre-download Docling's layout/table models into the image (otherwise the first DAG run does it).
RUN python -c "from docling.utils.model_downloader import download_models; download_models()" || \
    echo "Model pre-download skipped; Docling will download on first run"

COPY --chown=airflow:root airflow/start.sh /opt/airflow/start.sh
# Strip Windows line endings in case Git converted them.
RUN sed -i 's/\r$//' /opt/airflow/start.sh && chmod +x /opt/airflow/start.sh
