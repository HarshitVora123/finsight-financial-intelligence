FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/run_pipeline.py --customers 800
EXPOSE 8000 8501
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
