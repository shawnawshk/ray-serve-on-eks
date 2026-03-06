# Ray Serve + vLLM on EKS — Demo Design

## Context

Demonstrate Ray Serve's scaling capabilities to an engineering audience using EKS with a
Karpenter GPU nodepool already provisioned. The hero of the demo is Ray Serve autoscaling
and zero-downtime upgrades. vLLM is the inference backend engine; Karpenter GPU provisioning
is supporting infrastructure.

Model: `Qwen/Qwen2.5-2B` (small, fits in a single GPU, fast to load).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  EKS cluster (eks-networking)                           │
│                                                         │
│  ┌──────────────────┐   ┌───────────────────────────┐  │
│  │  KubeRay Operator│   │  Karpenter GPU NodePool   │  │
│  └────────┬─────────┘   └───────────────────────────┘  │
│           │ manages                    ↑ provisions      │
│  ┌────────▼─────────────────────────────────────────┐   │
│  │  RayService                                      │   │
│  │  ┌─────────────┐   ┌────────────────────────┐   │   │
│  │  │  Head node  │   │  Worker group (GPU)    │   │   │
│  │  │  (CPU)      │   │  replica 1..N          │   │   │
│  │  │  scheduler  │   │  vLLM + Qwen2.5-2B     │   │   │
│  │  │  dashboard  │   │  nvidia.com/gpu: 1     │   │   │
│  │  └─────────────┘   └────────────────────────┘   │   │
│  │                                                  │   │
│  │  Ray Serve: /v1/chat/completions (OpenAI API)    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         ↑ load test (locust)    ↑ observe (Ray Dashboard)
```

---

## Components

### 1. KubeRay Operator
- Installed via Helm into `ray-system` namespace
- Manages `RayService`, `RayCluster`, `RayJob` CRDs
- Standard install, no custom config

### 2. RayService (`k8s/ray/rayservice.yaml`)
- **Head node**: CPU-only (no GPU wasted on scheduling). Runs Ray GCS, dashboard,
  Serve controller. Scheduled on existing CPU nodes.
- **Worker group**: `minReplicas: 1`, `maxReplicas: 4`. Each worker requests
  `nvidia.com/gpu: 1` and tolerates the `nvidia.com/gpu:NoSchedule` taint.
  Karpenter provisions a GPU node per worker replica.
- **serveConfigV2**: inline Serve application config pointing to `serve.app:deployment`.
  Runtime env declares `pip: [vllm]` — no custom image build required.
- **Upgrade strategy**: RayService keeps the old cluster alive and canary-routes
  traffic to the new cluster during `kubectl apply` updates. Zero dropped requests,
  visible in the dashboard.

### 3. Ray Serve Application (`serve/app.py`)
- `VLLMDeployment` class decorated with `@serve.deployment` and `@serve.ingress(FastAPI)`
- Uses `AsyncLLMEngine` for proper concurrent request handling within a replica
- Autoscaling config: `min_replicas=1`, `max_replicas=4`, `target_ongoing_requests=5`
- Exposes OpenAI-compatible `/v1/chat/completions` endpoint
- `ray_actor_options={"num_gpus": 1}` — 1 replica = 1 GPU = 1 Karpenter node

### 4. Load Test (`load-test/locustfile.py`)
- Locust hitting `/v1/chat/completions` with concurrent users
- Drives queue depth above `target_ongoing_requests` threshold to trigger autoscaling
- Run alongside Ray Dashboard to make the scaling visible

---

## Demo Flow

1. `helm install kuberay-operator` → `kubectl apply -f k8s/ray/rayservice.yaml`
2. `kubectl get nodes -w` — watch Karpenter provision the first GPU node
3. `kubectl get rayservice -w` — watch cluster form and model load
4. `curl /v1/chat/completions` — confirm the endpoint responds
5. `locust -f load-test/locustfile.py` — ramp up load
6. Open Ray Dashboard — watch replicas scale 1→4, Karpenter add GPU nodes
7. `kubectl apply` with a config change — Dashboard shows canary rollout, no downtime

---

## File Layout

```
ray-serve-demo/
  k8s/
    karpenter/karpenter.yaml       # GPU EC2NodeClass + NodePool (done)
    nvidia/device-plugin.yaml      # NVIDIA device plugin DaemonSet (done)
    kuberay/
      operator-values.yaml         # Helm values for KubeRay operator
    ray/
      rayservice.yaml              # RayService manifest
  serve/
    app.py                         # Ray Serve + vLLM application
  load-test/
    locustfile.py                  # Locust load test
  docs/plans/
    2026-03-06-ray-serve-vllm-design.md
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| KubeRay CRD | `RayService` | Owns full lifecycle + canary upgrades |
| Inference engine | vLLM `AsyncLLMEngine` | Handles concurrency within a replica correctly |
| Image strategy | `rayproject/ray:2.44.0-gpu` + runtime pip | No image build step, fast iteration |
| Autoscaling trigger | `target_ongoing_requests: 5` | Easy to trigger with locust, visible in dashboard |
| API surface | OpenAI-compatible `/v1/chat/completions` | Works with any client, no custom tooling |
| GPU per replica | 1 | 1:1 with Karpenter nodes, clean scaling story |
