FROM python:3.12-slim

WORKDIR /zsl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "zsl_server:app", "--host", "0.0.0.0", "--port", "8001"]
