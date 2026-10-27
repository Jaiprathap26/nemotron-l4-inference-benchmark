FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx pytest
COPY app/ ./app/
COPY tests/ ./tests/
EXPOSE 8080
CMD ["uvicorn", "app.thinking_router:app", "--host", "0.0.0.0", "--port", "8080"]
