# System Architecture - ESRGAN Image Enhancer

This document details the multi-tier engineering architecture, data-flows, database relationships, and concurrent background processing strategy implemented for the **ESRGAN Image Enhancer** platform.

---

## 1. High-Level Architecture Overview

The system is split into three main isolated operational tiers:

```mermaid
graph TD
    User([End User / Web Browser]) -->|HTTPS / WSS| Frontend[React Single Page Application]
    Frontend -->|REST API Requests| Gateway[FastAPI Web Server]
    
    subgraph backend [Backend API Tier]
        Gateway -->|Asynchronous Controller| DB[(SQLite Database via aiosqlite)]
        Gateway -->|Enqueue Task| Redis{Redis Task Broker}
        Gateway -->|Fallback Local Thread| ThreadPool[ThreadPoolExecutor Inference]
    end

    subgraph tasks [ML Processing Worker Tier]
        Redis -->|Task Dispatch| Worker[Celery Processing Worker]
        Worker -->|Load Image & Model| InferenceEngine[ESRGAN PyTorch/ONNX Runtime Engine]
        InferenceEngine -->|Upscaled Result| OutputStorage[(Local Safe Sandbox Storage)]
        Worker -->|Update Status| DB
    end
    
    ThreadPool -->|Run In-Process ML| InferenceEngine
```

---

## 2. Tier Components

### A. Frontend Tier (Client Dashboard)
- **Framework**: React 18, Vite (for hot module reloading and high-performance compilation), and TypeScript (strict types).
- **Global State Store**: Zustand (zero-boilerplate, high-performance state store to decouple task state and theme variables).
- **Core Interactions**: Drag-and-drop file inputs (react-dropzone), dynamic progress tracking trackers, and custom coordinate-tracking before/after slider inspection layer.

### B. Backend API Tier (FastAPI Gateway)
- **Web Application**: FastAPI running on Uvicorn. Exposes REST endpoints for image enhancements, task polling, cancellation, and artifact delivery.
- **Task Scheduling Manager**: Decouples network request thread pools from intensive ML inferences using a dual-orchestration path:
  1. **Celery Worker Fallback**: Standard configuration routes tasks to an active Redis broker for queue scheduling.
  2. **Thread Pool Executor**: If Redis is offline, the gateway utilizes safe background thread executors to process local scaling without dropping client uploads.
- **Relational Storage**: Async SQLite session database pools via SQLAlchemy and `aiosqlite` tracking metadata for historical auditing.

### C. Processing Worker Tier (Celery + PyTorch)
- **Worker Process**: Single or multi-container Celery processing environments subscribed to incoming Redis queues.
- **Super-Resolution Engine**: Houses the generator network. Supports both direct PyTorch weights (`.pth`) and GPU-accelerated ONNX runtimes.
- **Safety Pruner**: Multi-threaded periodic files cleaner that scrubs upload sandboxes and upscaled artifacts older than 24 hours to enforce absolute data privacy.

---

## 3. Detailed Data Flow Sequence

The diagram below maps the end-to-end lifecycle of an image enhancement request:

```mermaid
sequenceDiagram
    autonumber
    actor User as Client (Web Dashboard)
    participant API as FastAPI Gateway
    participant DB as SQLite DB
    participant MQ as Redis Queue
    participant CW as Celery Worker
    participant Disk as Local Sandbox Disk

    User->>API: Upload File (Multipart Form) + Scale Factor (2x/4x/8x)
    Note over API: Run image size, format &<br/>magic signature validations
    API->>DB: Scaffold Task Status ("queued")
    API->>MQ: Enqueue Enhancement Task Payload
    API-->>User: Return HTTP 202 (Task ID + Estimated Duration)
    
    par Background Status Polling
        loop Every 2 seconds
            User->>API: GET /api/v1/task/{taskId}
            API->>DB: Fetch Task Status
            DB-->>API: Task Record
            API-->>User: Status + Progress Percentage
        end
    and Task Worker Execution
        MQ->>CW: Pull Queue Task Payload
        CW->>DB: Update Status to "processing" (progress = 10%)
        CW->>Disk: Load Input Image
        CW->>Disk: Run PyTorch Inference (Generator Network)
        Disk-->>CW: Enhancement Complete (Save Output File)
        CW->>DB: Update Status to "completed" (progress = 100%)
    end

    User->>API: Status is "completed", GET /api/v1/result/{taskId}
    API->>Disk: Fetch Artifact from safe directory
    Disk-->>API: Raw Image Stream
    API-->>User: File Stream Download (Attachment)
```

---

## 4. Database Schema Design

The SQLite database tracks execution metrics. Below is the relational entity model:

```mermaid
classDiagram
    class EnhancementTask {
        +String task_id (PK)
        +String input_filename
        +String input_path
        +String output_path
        +Integer scale_factor
        +String status
        +Integer progress
        +Float processing_time
        +String error_message
        +DateTime created_at
        +DateTime updated_at
    }
```

- **Index on Task Status**: An index is applied to the `status` column to accelerate status lookups.
- **Cascading Audits**: Every state shift automatically mutates the `updated_at` timestamps using SQLite hooks.
