# Ray Serve + vLLM on EKS

Scalable LLM inference on Amazon EKS using [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) and [vLLM](https://docs.vllm.ai/), with GPU autoscaling powered by [Karpenter](https://karpenter.sh/).

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  EKS Cluster                                                  │
│                                                               │
│  ┌────────────────┐          ┌──────────────────────────────┐ │
│  │ KubeRay        │ manages  │ RayService                   │ │
│  │ Operator       │─────────▶│                              │ │
│  └────────────────┘          │  ┌──────────┐ ┌───────────┐  │ │
│                              │  │ Head     │ │ Worker(s) │  │ │
│  ┌────────────────┐          │  │ (CPU)    │ │ (GPU)     │  │ │
│  │ Karpenter      │provisions│  │ GCS      │ │ vLLM      │  │ │
│  │ GPU NodePool   │◀─────────│  │ Dashboard│ │ Engine    │  │ │
│  └────────────────┘    nodes │  │ Proxy    │ │ 1 GPU/rep │  │ │
│                              │  └──────────┘ └───────────┘  │ │
│  ┌────────────────┐          │                              │ │
│  │ NVIDIA GPU     │          │  OpenAI API: /v1/chat/...    │ │
│  │ Operator       │          └──────────────────────────────┘ │
│  └────────────────┘                                           │
└───────────────────────────────────────────────────────────────┘
         ▲ load test (Locust)         ▲ observe (Ray Dashboard)
```

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Serving framework | Ray Serve `RayService` CRD | Full lifecycle management + zero-downtime canary upgrades |
| Inference engine | vLLM native OpenAI serving layer | Streaming, tool calling, `/v1/models`, `/v1/completions` out of the box |
| Code packaging | **ConfigMap** (not baked into image) | Runtime decoupled from app logic — update code without rebuilding the image |
| Autoscaling config | In `serveConfigV2` (not in Python) | Single source of truth, declarative, `kubectl apply` to change |
| GPU per replica | 1 | Clean 1:1 mapping with Karpenter nodes |
| Image | `rayproject/ray:2.54.0-py312-gpu` + `pip install vllm` | Minimal custom image, latest vLLM compatibility |

## Project Structure

```
ray-serve-demo/
├── Dockerfile                          # Runtime image (Ray + vLLM, no app code)
├── docker-compose.yaml                 # Local dev cluster (CPU, no GPU)
├── serve/
│   ├── vllm_serve.py                   # vLLM OpenAI serving deployment (source of truth)
│   ├── app.py                          # Legacy hand-rolled serve app (reference)
│   ├── dummy_app.py                    # Minimal app for local testing
│   └── wait_and_test.py                # Health check helper
├── k8s/
│   ├── ray/
│   │   ├── rayservice.yaml             # RayService manifest
│   │   ├── vllm-serve-configmap.yaml   # ConfigMap (generated from serve/vllm_serve.py)
│   │   └── open-webui.yaml             # Open WebUI frontend (Deployment + ClusterIP Service)
│   ├── karpenter/
│   │   └── gpu-nodepool.yaml           # GPU NodePool + EC2NodeClass
│   ├── kuberay/
│   │   └── operator-values.yaml        # KubeRay operator Helm values
│   └── gpu-operator/
│       └── values.yaml                 # NVIDIA GPU Operator Helm values
├── load-test/
│   └── locustfile.py                   # Locust load test
└── docs/plans/
    └── *.md                            # Design docs
```

## Prerequisites

- EKS cluster with:
  - [KubeRay Operator](https://ray-project.github.io/kuberay/) installed
  - [Karpenter](https://karpenter.sh/) with a GPU NodePool
  - [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/) (driver disabled if using GPU-optimized AMI)
- ECR repository for the runtime image
- HuggingFace token (stored as K8s Secret `hf-token`)

## Quick Start

### 1. Build and push the runtime image

```bash
# Login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-west-2.amazonaws.com

# Build and push (no app code in image)
docker buildx build --platform linux/amd64 \
  -t <ACCOUNT>.dkr.ecr.us-west-2.amazonaws.com/ray-serve-vllm:latest \
  --push .
```

### 2. Create the HuggingFace token secret

```bash
kubectl create secret generic hf-token --from-literal=HF_TOKEN=hf_xxxxx
```

### 3. Deploy

```bash
# Apply the serve script as ConfigMap
kubectl apply -f k8s/ray/vllm-serve-configmap.yaml

# Deploy the RayService
kubectl apply -f k8s/ray/rayservice.yaml
```

### 4. Watch it come up

```bash
# Watch pods
kubectl get pods -w

# Check Ray Serve status
kubectl exec <head-pod> -c ray-head -- serve status
```

### 5. Test

```bash
kubectl port-forward svc/vllm-serve-serve-svc 8000:8000

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-2B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### 6. (Optional) Open WebUI

Deploy [Open WebUI](https://github.com/open-webui/open-webui) as a chat frontend to verify the model interactively:

```bash
kubectl apply -f k8s/ray/open-webui.yaml
kubectl port-forward svc/open-webui 8080:80
# Open http://localhost:8080
```

Open WebUI auto-discovers models from the vLLM backend (`/v1/models`). Auth is disabled for demo use.

## Updating the Serve App

The application code lives in a ConfigMap, **not in the container image**. To update:

```bash
# Edit serve/vllm_serve.py locally, then regenerate the ConfigMap:
kubectl create configmap vllm-serve-script \
  --from-file=vllm_serve.py=serve/vllm_serve.py \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart the RayService to pick up the new code
kubectl delete rayservice vllm-serve && kubectl apply -f k8s/ray/rayservice.yaml
```

No image rebuild needed.

## Changing the Model

Edit the `MODEL_ID` env var in `k8s/ray/rayservice.yaml` under `serveConfigV2.applications[0].runtime_env.env_vars`:

```yaml
env_vars:
  MODEL_ID: "Qwen/Qwen3.5-2B"    # ← change this
  MAX_MODEL_LEN: "4096"
  TENSOR_PARALLEL_SIZE: "1"
```

Then `kubectl apply -f k8s/ray/rayservice.yaml`. RayService will canary-deploy the new config with zero downtime.

## Autoscaling

Autoscaling is configured in `serveConfigV2`:

```yaml
autoscaling_config:
  min_replicas: 1
  max_replicas: 4
  target_num_ongoing_requests_per_replica: 5
  upscale_delay_s: 30
  downscale_delay_s: 600
```

Each replica requires 1 GPU. When load exceeds the target, Ray Serve adds replicas → Karpenter provisions new GPU nodes → new replicas start serving. Scale-down waits 10 minutes after load drops.

## Load Testing

```bash
pip install locust
locust -f load-test/locustfile.py --host http://localhost:8000
```

Open the Ray Dashboard (`kubectl port-forward svc/vllm-serve-head-svc 8265:8265`) to watch replicas scale in real time.

## Observability

| Endpoint | Description |
|---|---|
| `:8000/health` | Serve app health check |
| `:8000/metrics` | Prometheus metrics (vLLM + Ray Serve) |
| `:8265` | Ray Dashboard |

## Stack

| Component | Version |
|---|---|
| Ray | 2.54.0 |
| vLLM | 0.17.0 |
| KubeRay Operator | latest |
| Model | Qwen/Qwen3.5-2B |
| Python | 3.12 |

## Local Development (CPU, no GPU)

For testing the Ray Serve pipeline without GPUs:

```bash
docker compose up
# Deploys dummy_app.py (echo server) on a 2-node Ray cluster
# Dashboard: http://localhost:8265
# Serve API: http://localhost:8000
```
