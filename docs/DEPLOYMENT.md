# Deployment Guide - ESRGAN Image Enhancer

This document provides step-by-step instructions for deploying the **ESRGAN Image Enhancer** platform to both local developer machines and production cloud server infrastructure (AWS ECS, Docker environments, or CPU-only web services).

---

## 1. Environment Configurations Matrix

Create a `.env` file based on `.env.example` in the root of the project:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker URL. If offline, the server falls back to thread pools. |
| `DATABASE_URL` | `sqlite+aiosqlite:///./esrgan.db` | Async SQLAlchemy SQLite database path. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS trusted clients. Comma-separated list. |
| `MAX_FILE_SIZE` | `10485760` | Maximum file upload size in bytes (10MB). |
| `ENABLE_CUDA` | `true` | Set to `true` to utilize active GPUs if CUDA is configured. |
| `MODEL_PRECISION` | `fp32` | Weights precision mode: `fp16` or `fp32`. |
| `UPLOAD_DIR` | `data/uploads` | Safe sandboxed local input storage directory. |
| `RESULT_DIR` | `data/results` | Safe sandboxed local output storage directory. |

---

## 2. Docker Compose Deployment (Recommended for Production)

Docker Compose coordinates the backend API gateway, the Celery queue dispatcher, the Redis in-memory broker, and the static Nginx frontend web server in complete isolation.

### A. Non-GPU / CPU-Only Deployments
To run the containers using only the CPU (suitable for lightweight servers, testing, or development):

```bash
# 1. Boot up the services in detached mode
docker-compose up --build -d

# 2. Inspect active container statuses
docker-compose ps

# 3. Stream backend logs to verify database table creation
docker-compose logs -f backend
```

Once online:
- **Frontend Dashboard**: Navigate to [http://localhost:3000](http://localhost:3000)
- **FastAPI OpenAPI Swagger**: Interact at [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**: Ping [http://localhost:8000/health](http://localhost:8000/health)

### B. Hardware Accelerated GPU Deployments (AWS/Scyld/CoreWeave)
If you have an NVIDIA CUDA compatible graphics card and have installed the NVIDIA Container Toolkit on your host OS:

1. Enable GPU pass-through inside the backend service under `docker-compose.yml`:
   ```yaml
   backend:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: all
               capabilities: [gpu]
   ```
2. Run the compose services:
   ```bash
   ENABLE_CUDA=true docker-compose up --build -d
   ```

---

## 3. Local Developer Setup (No Container Isolation)

If you need to run, debug, or tweak the project directly on a Windows or Linux machine without containers:

### Prerequisites
- **Python**: v3.10 to v3.13
- **Node.js**: v20 or later
- **Redis**: An active Redis server running locally on standard port `6379`.

### A. Backend Scaffolding
```bash
# 1. Navigate to backend directory
cd backend

# 2. Create a clean virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

# 3. Install core backend packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download weights or mock placeholders
python ml/weights/download_weights.py

# 5. Boot FastAPI API server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In a separate terminal (with virtual environment active):
```bash
# 6. Boot the background Celery Task Worker
celery -A app.tasks.celery_app worker --loglevel=info -P threads
```

### B. Frontend Scaffolding
```bash
# 1. Navigate to frontend directory
cd ../frontend

# 2. Install dependencies
npm install

# 3. Spin up the Vite dev server
npm run dev
```

The frontend dashboard will run locally on [http://localhost:3000](http://localhost:3000).

---

## 4. Production Cloud Architectures

### A. Railway / Render / Fly.io (Platform as a Service)
You can deploy this application directly from Git using the Dockerfiles:
- **API Service**: Point Railway to [backend/Dockerfile](file:///c:/Users/fs885/Desktop/esrgan/esrgan-enhancer/backend/Dockerfile). Link it to a Redis Database plugin and set `REDIS_URL` in environment variables.
- **Worker Service**: Point Railway to the same backend Dockerfile, but overwrite the execution start command to:
  `celery -A app.tasks.celery_app worker --loglevel=info`
- **Client App**: Point Render or Railway to [frontend/Dockerfile](file:///c:/Users/fs885/Desktop/esrgan/esrgan-enhancer/frontend/Dockerfile) and hook up the static Nginx server directly.

### B. AWS ECS (Elastic Container Service) with GPUs
1. **Container Repository**: Push backend and frontend docker builds to AWS ECR (Elastic Container Registry).
2. **Cluster Creation**: Allocate an Amazon ECS cluster running on an EC2 launch type with a `g4dn` instance type (features NVIDIA T4 GPUs).
3. **Task Definition**: Specify a task definition mapping target GPU nodes. Add environment variables pointing to an active ElastiCache Redis cluster.
4. **Application Load Balancer**: Route domain traffic through an AWS ALB, directing `/api/*` to the FastAPI backend service and `/` or static files to the Nginx frontend task container.
