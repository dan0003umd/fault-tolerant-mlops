FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ARG CACHEBUST
RUN python generate_synthetic_log.py && python train.py
CMD ["uvicorn", "zsl_server:app", "--host", "0.0.0.0", "--port", "8001"]