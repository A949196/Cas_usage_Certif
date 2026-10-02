# Dockerfile du service frontend (interface conseiller Streamlit)
#
# Build : docker build -f services/frontend/Dockerfile -t trajectoire-emploi-ui:v0.1.0-ui .
# Run   : docker run -d -p 8501:8501 -e API_URL=http://backend:8000 trajectoire-emploi-ui:v0.1.0-ui

FROM python:3.11-slim

RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /home/appuser/app

COPY --chown=appuser:appuser services/frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser services/frontend/app.py ./app.py

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
