\# Fault-Tolerant ML Inference Service



This project is a small \*\*fault-tolerant MLOps lab\*\* that exposes a FastAPI microservice, injects faults into model inference, and monitors behavior using \*\*Prometheus\*\* and \*\*Grafana\*\*.



The goal is to simulate production-style failures (latency spikes, error bursts, etc.), observe them via metrics, and reason about recovery and SLOs.



\---



\## Architecture



The stack runs via Docker Compose with three services:



| Service            | Image                    | Port | Purpose                        |

|--------------------|--------------------------|------|--------------------------------|

| ftmlops-app        | fault-tolerant-mlops-app | 8000 | FastAPI ML inference service   |

| ftmlops-prometheus | prom/prometheus:latest   | 9090 | Metrics collection \& scraping  |

| ftmlops-grafana    | grafana/grafana:latest   | 3000 | Metrics visualization          |



\---



\## Repository structure



```text

fault-tolerant-mlops/

├── app/

│   ├── main.py              # FastAPI app with fault injection + logging + metrics

│   ├── train.py             # Model training script (runs at build time)

│   └── requirements.txt     # Python dependencies

├── loadgen/

│   ├── loadgen.py           # Load generator script

│   └── run\_all\_experiments.sh

├── monitoring/

│   └── prometheus.yml       # Prometheus scrape configuration

├── docker-compose.yml       # Orchestrates app, Prometheus, Grafana

└── README.md

```



\*\*FastAPI app (`app/main.py`) exposes:\*\*



\- `GET /health` – liveness / readiness check.

\- `POST /predict` – iris classifier prediction endpoint (returns class + latency).

\- `GET /metrics` – Prometheus metrics endpoint.

\- `GET /get\_mode` – read current fault injection mode.

\- `GET /set\_mode?mode=...` – set fault injection mode (e.g., `normal`, `error\_spike`, `slow`).



\---



\## Running locally with Docker Compose



Prerequisites:



\- Docker and Docker Compose.



Steps:



```bash

git clone https://github.com/dan0003umd/fault-tolerant-mlops.git

cd fault-tolerant-mlops



\# Build and start all services in the background

docker compose up -d --build

```



Services (default ports):



\- FastAPI: `http://localhost:8000`

\- FastAPI docs: `http://localhost:8000/docs`

\- FastAPI metrics: `http://localhost:8000/metrics`

\- Prometheus: `http://localhost:9090`

\- Grafana: `http://localhost:3000`



To stop everything:



```bash

docker compose down

```



\---



\## Health \& quick checks



Once `docker compose up -d --build` completes, verify:



1\. \*\*App health\*\*



&#x20;  ```bash

&#x20;  curl http://localhost:8000/health

&#x20;  ```



&#x20;  Expected:



&#x20;  ```json

&#x20;  {"status": "ok"}

&#x20;  ```



2\. \*\*OpenAPI docs\*\*



&#x20;  - Browser: `http://localhost:8000/docs`



3\. \*\*Metrics endpoint\*\*



&#x20;  ```bash

&#x20;  curl http://localhost:8000/metrics

&#x20;  ```



&#x20;  You should see Prometheus-formatted metrics.



4\. \*\*Docker Compose status\*\*



&#x20;  ```bash

&#x20;  docker compose ps

&#x20;  ```



&#x20;  Expect:



&#x20;  - `ftmlops-app` – Up

&#x20;  - `ftmlops-prometheus` – Up

&#x20;  - `ftmlops-grafana` – Up



5\. \*\*Prometheus UI\*\*



&#x20;  - Browser: `http://localhost:9090`

&#x20;  - Go to `Status → Targets` and confirm the FastAPI target is \*\*UP\*\*.



6\. \*\*Grafana UI\*\*



&#x20;  - Browser: `http://localhost:3000`

&#x20;  - Configure Prometheus as a data source and open dashboards for:

&#x20;    - Request rate

&#x20;    - Error rate

&#x20;    - Latency

&#x20;    - Total requests



\---



\## Fault injection modes



The service supports three runtime fault modes, switchable without restarting:



| Mode        | Behavior                                    | How to enable (example)                                      |

|-------------|---------------------------------------------|--------------------------------------------------------------|

| `normal`    | Standard inference, no injected faults      | `curl "http://localhost:8000/set\_mode?mode=normal"`         |

| `error\_spike` | \~30% of requests return HTTP 500          | `curl "http://localhost:8000/set\_mode?mode=error\_spike"`    |

| `slow`      | \~1.5 second delay injected before response | `curl "http://localhost:8000/set\_mode?mode=slow"`           |



Current mode:



```bash

curl http://localhost:8000/get\_mode

```



\---



\## Metrics



At `/metrics`, the app exposes custom Prometheus metrics including:



\- `app\_requests\_total` – total prediction requests received.

\- `app\_prediction\_errors\_total` – total failed predictions.

\- `app\_request\_latency\_seconds` – request latency histogram.



Example PromQL (in Grafana or Prometheus):



\- Request rate:



&#x20; ```promql

&#x20; sum(rate(app\_requests\_total\[1m]))

&#x20; ```



\- Error rate:



&#x20; ```promql

&#x20; sum(rate(app\_prediction\_errors\_total\[1m]))

&#x20; ```



\- Average latency:



&#x20; ```promql

&#x20; rate(app\_request\_latency\_seconds\_sum\[1m])

&#x20;   / rate(app\_request\_latency\_seconds\_count\[1m])

&#x20; ```



\---



\## Load generation \& experiments



From inside the repo:



```bash

cd loadgen



\# Simple run: 60 seconds at 10 RPS in normal mode

python loadgen.py --rate 10 --duration 60 --mode normal



\# Full scripted experiment suite (multiple load levels \& modes)

bash run\_all\_experiments.sh

```



During experiments:



\- Prometheus records all metrics.

\- Grafana dashboards can be used to observe:

&#x20; - Normal vs. high vs. critical load.

&#x20; - Error spike behavior.

&#x20; - Latency spikes in `slow` mode.

&#x20; - Total request counts over time.



\---



\## Deployment on EC2 (current setup)



This repo is deployed on an AWS EC2 instance.



\- EC2 public IP: `3.141.165.28`

\- Example URLs (when the stack is running):



&#x20; - FastAPI docs: `http://3.141.165.28:8000/docs`

&#x20; - Health: `http://3.141.165.28:8000/health`

&#x20; - Metrics: `http://3.141.165.28:8000/metrics`

&#x20; - Prometheus: `http://3.141.165.28:9090`

&#x20; - Grafana: `http://3.141.165.28:3000`



To update the EC2 deployment from GitHub:



```bash

ssh ubuntu@3.141.165.28



cd \~/fault-tolerant-mlops

git pull

docker compose up -d --build

```



\---



\## Team notes



\- Repo owner: `dan0003umd`

\- This repository is the \*\*source of truth\*\*; EC2 pulls from the `main` branch.

\- Any teammate can:

&#x20; - Clone and run locally with Docker Compose.

&#x20; - Or hit the live EC2 endpoints (if their IP is allowed in the security group).

