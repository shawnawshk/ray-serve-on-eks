# Ray Serve + vLLM on EKS — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy Qwen2.5-2B via vLLM + Ray Serve on EKS using KubeRay, then demonstrate autoscaling and zero-downtime upgrades to an engineering audience.

**Architecture:** KubeRay operator manages a `RayService` CRD that owns a head node (CPU) and a GPU worker group. Each GPU worker runs one vLLM `AsyncLLMEngine` replica exposed as an OpenAI-compatible Ray Serve deployment. Karpenter provisions GPU nodes on demand per worker replica. App code is delivered via ConfigMap mounted into both head and worker pods.

**Tech Stack:** KubeRay 1.x (Helm), Ray 2.44.0-gpu, vLLM (latest via runtime pip), FastAPI, Locust, kubectl, Helm 3

---

## Task 1: Install KubeRay Operator

**Files:**
- Create: `k8s/kuberay/operator-values.yaml`

**Step 1: Create Helm values file**

```yaml
# k8s/kuberay/operator-values.yaml
resources:
  limits:
    cpu: "1"
    memory: 512Mi
  requests:
    cpu: "250m"
    memory: 256Mi
```

**Step 2: Add KubeRay Helm repo and install**

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator \
  --namespace ray-system \
  --create-namespace \
  --version 1.3.0 \
  -f k8s/kuberay/operator-values.yaml
```

**Step 3: Verify operator is running**

```bash
kubectl get pods -n ray-system
```

Expected: one `kuberay-operator-*` pod in `Running` state.

**Step 4: Verify CRDs are installed**

```bash
kubectl get crd | grep ray
```

Expected: `rayclusters.ray.io`, `rayservices.ray.io`, `rayjobs.ray.io` all present.

**Step 5: Commit**

```bash
git add k8s/kuberay/operator-values.yaml
git commit -m "add kuberay operator helm values"
```

---

## Task 2: Write the Ray Serve + vLLM Application

**Files:**
- Create: `serve/app.py`

**Step 1: Write `serve/app.py`**

```python
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

fastapi_app = FastAPI()

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-2B")


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 5,
        "upscaling_factor": 1.0,
        "downscale_delay_s": 60,
    },
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=20,
)
@serve.ingress(fastapi_app)
class VLLMDeployment:
    def __init__(self):
        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=2048,
            dtype="auto",
            enforce_eager=True,  # skip CUDA graph compile — faster cold start for demo
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @fastapi_app.get("/health")
    async def health(self):
        return {"status": "ok"}

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(self, request: dict):
        messages = request.get("messages", [])
        # Flatten messages to a single prompt string
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        ) + "\nassistant:"

        sampling_params = SamplingParams(
            temperature=float(request.get("temperature", 0.7)),
            max_tokens=int(request.get("max_tokens", 256)),
            stop=["user:", "\n\n"],
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        final_output = None
        async for output in results_generator:
            final_output = output

        if final_output is None:
            return JSONResponse(status_code=500, content={"error": "no output"})

        text = final_output.outputs[0].text.strip()
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }


deployment = VLLMDeployment.bind()
```

**Step 2: Confirm the file is syntactically valid**

```bash
python3 -c "import ast; ast.parse(open('serve/app.py').read()); print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add serve/app.py
git commit -m "add ray serve vllm deployment app"
```

---

## Task 3: Create ConfigMap for Serve App

The app code is delivered to pods via a Kubernetes ConfigMap mounted at
`/home/ray/workspace/`. PYTHONPATH on the pods includes that path, so
`import_path: app:deployment` resolves correctly.

**Files:**
- Generated: ConfigMap in cluster (source of truth is `serve/app.py`)

**Step 1: Create the ConfigMap from the file**

```bash
kubectl create configmap serve-app \
  --from-file=app.py=serve/app.py \
  --dry-run=client -o yaml | kubectl apply -f -
```

Expected: `configmap/serve-app created` (or `configured` on re-apply).

**Step 2: Verify the ConfigMap contains the file**

```bash
kubectl get configmap serve-app -o jsonpath='{.data.app\.py}' | head -5
```

Expected: First lines of `serve/app.py`.

> **Note:** Whenever `serve/app.py` changes, re-run Step 1 to refresh the ConfigMap,
> then trigger a RayService rollout (Task 5, Step 5).

---

## Task 4: Write the RayService Manifest

**Files:**
- Create: `k8s/ray/rayservice.yaml`

**Step 1: Write `k8s/ray/rayservice.yaml`**

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: vllm-serve
  namespace: default
spec:
  # Time before RayService is declared unhealthy during initial deploy
  serviceUnhealthySecondThreshold: 600   # model load can take a few minutes
  deploymentUnhealthySecondThreshold: 600

  serveConfigV2: |
    applications:
    - name: vllm-app
      import_path: app:deployment
      route_prefix: /
      runtime_env:
        pip:
          - vllm
        env_vars:
          MODEL_ID: "Qwen/Qwen2.5-2B"

  rayClusterConfig:
    enableInTreeAutoscaling: true
    autoscalerOptions:
      upscalingMode: Default
      resources:
        requests:
          cpu: "500m"
          memory: "512Mi"
        limits:
          cpu: "500m"
          memory: "512Mi"

    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
        num-cpus: "0"        # prevent tasks/actors scheduling on head
      template:
        spec:
          nodeSelector:
            node-type: cpu   # keep head off GPU nodes (adjust if needed)
          containers:
          - name: ray-head
            image: rayproject/ray:2.44.0-gpu
            ports:
            - containerPort: 6379   # GCS
            - containerPort: 8265   # Dashboard
            - containerPort: 8000   # Serve HTTP
            resources:
              requests:
                cpu: "2"
                memory: "8Gi"
              limits:
                cpu: "2"
                memory: "8Gi"
            env:
            - name: PYTHONPATH
              value: /home/ray/workspace
            volumeMounts:
            - name: serve-app
              mountPath: /home/ray/workspace
          volumes:
          - name: serve-app
            configMap:
              name: serve-app

    workerGroupSpecs:
    - groupName: gpu-worker
      replicas: 1
      minReplicas: 1
      maxReplicas: 4
      rayStartParams: {}
      template:
        spec:
          tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
          nodeSelector:
            node-type: gpu   # target Karpenter GPU nodes
          containers:
          - name: ray-worker
            image: rayproject/ray:2.44.0-gpu
            resources:
              requests:
                cpu: "4"
                memory: "16Gi"
                nvidia.com/gpu: "1"
              limits:
                cpu: "4"
                memory: "16Gi"
                nvidia.com/gpu: "1"
            env:
            - name: PYTHONPATH
              value: /home/ray/workspace
            - name: HF_HOME
              value: /tmp/hf_cache   # model cache inside container
            volumeMounts:
            - name: serve-app
              mountPath: /home/ray/workspace
          volumes:
          - name: serve-app
            configMap:
              name: serve-app
```

> **Note on `node-type: cpu` nodeSelector on head:** The existing default
> Karpenter NodePool covers CPU nodes without that label. Either remove the
> nodeSelector on the head, or add `node-type: cpu` label to the default
> NodePool. Simplest for the demo: remove the head nodeSelector entirely
> and rely on the GPU taint to repel the head pod from GPU nodes.

**Step 2: Remove head nodeSelector (simpler for demo)**

The GPU taint (`nvidia.com/gpu:NoSchedule`) already prevents the head pod
from landing on GPU nodes since it has no toleration. Remove the
`nodeSelector` block from `headGroupSpec.template.spec` entirely.

Final head spec (no nodeSelector):
```yaml
headGroupSpec:
  rayStartParams:
    dashboard-host: "0.0.0.0"
    num-cpus: "0"
  template:
    spec:
      containers:
      - name: ray-head
        image: rayproject/ray:2.44.0-gpu
        ports:
        - containerPort: 6379
        - containerPort: 8265
        - containerPort: 8000
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "2"
            memory: "8Gi"
        env:
        - name: PYTHONPATH
          value: /home/ray/workspace
        volumeMounts:
        - name: serve-app
          mountPath: /home/ray/workspace
      volumes:
      - name: serve-app
        configMap:
          name: serve-app
```

**Step 3: Commit**

```bash
git add k8s/ray/rayservice.yaml
git commit -m "add rayservice manifest for vllm qwen2.5-2b"
```

---

## Task 5: Deploy and Verify

**Step 1: Apply the NVIDIA device plugin**

```bash
kubectl apply -f k8s/nvidia/device-plugin.yaml
```

Expected: `daemonset.apps/nvidia-device-plugin-daemonset created`

**Step 2: Apply the RayService**

```bash
kubectl apply -f k8s/ray/rayservice.yaml
```

Expected: `rayservice.ray.io/vllm-serve created`

**Step 3: Watch Karpenter provision a GPU node**

```bash
kubectl get nodes -w -l node-type=gpu
```

Expected: Within ~2 minutes, a new GPU node appears with `Ready` status.

**Step 4: Watch the RayService come up**

```bash
kubectl get rayservice vllm-serve -w
```

Wait until `STATUS` shows `Running`. This takes 5-10 minutes on first deploy
(vLLM pip install + model download from HuggingFace).

To follow progress, tail the worker pod logs:
```bash
kubectl logs -l ray.io/group=gpu-worker -f
```

**Step 5: Confirm the endpoint responds**

```bash
# Get the Serve head pod name
HEAD_POD=$(kubectl get pod -l ray.io/node-type=head -o name | head -1)

# Port-forward the serve port
kubectl port-forward $HEAD_POD 8000:8000 &

# Send a test request
curl -s http://localhost:8000/health
# Expected: {"status":"ok"}

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 50}'
```

Expected: JSON response with `choices[0].message.content` containing a short reply.

---

## Task 6: Write the Load Test

**Files:**
- Create: `load-test/locustfile.py`

**Step 1: Write `load-test/locustfile.py`**

```python
import json
import random
from locust import HttpUser, task, between

PROMPTS = [
    "Explain Kubernetes in one sentence.",
    "What is Ray Serve used for?",
    "Describe how autoscaling works.",
    "What is vLLM?",
    "How does Karpenter provision nodes?",
    "Explain GPU memory management briefly.",
    "What is the difference between Ray and Dask?",
    "Describe the OpenAI chat completions API format.",
]


class LLMUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def chat(self):
        payload = {
            "messages": [
                {"role": "user", "content": random.choice(PROMPTS)}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            catch_response=True,
            timeout=60,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"status {response.status_code}: {response.text[:200]}")
```

**Step 2: Commit**

```bash
git add load-test/locustfile.py
git commit -m "add locust load test for ray serve endpoint"
```

---

## Task 7: Demo — Scale-Up Under Load

This task is the live demo sequence, not code. Run these commands in order
with Ray Dashboard visible.

**Step 1: Open Ray Dashboard in browser**

```bash
kubectl port-forward $HEAD_POD 8265:8265 &
open http://localhost:8265
```

Navigate to **Serve** tab. You should see 1 replica of `VLLMDeployment`.

**Step 2: Install Locust and start load test**

```bash
pip install locust
locust -f load-test/locustfile.py \
  --host http://localhost:8000 \
  --headless \
  --users 20 \
  --spawn-rate 2 \
  --run-time 5m
```

**Step 3: Observe in Ray Dashboard**

- **Serve tab**: Replica count climbs from 1 → 2 → 4 as queue depth exceeds `target_ongoing_requests: 5`
- **Cluster tab**: New worker nodes appear as Karpenter provisions GPU instances
- `kubectl get nodes -w` in a separate terminal shows new GPU nodes joining

**Step 4: Stop load test, observe scale-down**

After ~60 seconds of idle (configured via `downscale_delay_s: 60`), replicas
drop back to 1 and Karpenter terminates the extra GPU nodes.

---

## Task 8: Demo — Zero-Downtime Upgrade

**Step 1: While traffic is running, change a config parameter**

Edit `k8s/ray/rayservice.yaml` — change `max_tokens` default or bump `max_replicas` to 6:

```yaml
# In serveConfigV2, change:
env_vars:
  MODEL_ID: "Qwen/Qwen2.5-2B"
  MAX_TOKENS: "512"          # add this new env var
```

**Step 2: Refresh the ConfigMap if app.py changed, then apply**

```bash
kubectl apply -f k8s/ray/rayservice.yaml
```

**Step 3: Watch the canary rollout in Ray Dashboard**

RayService spins up a new RayCluster alongside the old one. Traffic is
progressively shifted. Old cluster stays alive until the new one is healthy.

```bash
kubectl get rayservice vllm-serve -w
```

You will see `STATUS` transition through:
`Running` → `WaitForServeDeploymentReady` → `Running`

No requests are dropped during this transition.

---

## Troubleshooting Reference

| Symptom | Check |
|---|---|
| Worker pod stuck `Pending` | `kubectl describe pod <worker>` — check GPU taint toleration and Karpenter nodepool |
| vLLM pip install slow | Normal — takes 3-5 min. Watch with `kubectl logs -f <worker>` |
| Model download slow | `HF_HOME` is ephemeral; consider mounting an EFS PVC for caching between restarts |
| `RayService` stays `FailedToGetOrCreateRayCluster` | Check RBAC — KubeRay operator needs ClusterRole to create RayClusters |
| Head pod on GPU node | Remove head `nodeSelector`; GPU taint should repel it automatically |
| Serve endpoint returns 404 | Port-forward targets head pod port 8000, not the K8s Service. Check the head pod is the right one |
