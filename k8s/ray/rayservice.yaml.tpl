apiVersion: ray.io/v1
kind: RayService
metadata:
  name: vllm-serve
  namespace: default
spec:
  serviceUnhealthySecondThreshold: 1800
  deploymentUnhealthySecondThreshold: 1800

  serveConfigV2: |
    applications:
    - name: vllm-app
      import_path: "vllm_serve:deployment"
      route_prefix: /
      runtime_env:
        env_vars:
          MODEL_ID: "Qwen/Qwen3.5-2B"
          MAX_MODEL_LEN: "4096"
          MAX_NUM_SEQS: "32"
          TENSOR_PARALLEL_SIZE: "1"
          ENFORCE_EAGER: "true"
          GPU_MEMORY_UTILIZATION: "0.9"
      deployments:
        - name: vllm-deployment
          autoscaling_config:
            metrics_interval_s: 0.2
            min_replicas: 1
            max_replicas: 4
            look_back_period_s: 2
            downscale_delay_s: 600
            upscale_delay_s: 30
            target_num_ongoing_requests_per_replica: 5
          graceful_shutdown_timeout_s: 5
          max_concurrent_queries: 100
          ray_actor_options:
            num_cpus: 0
            num_gpus: 1

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
        num-cpus: "0"
      template:
        spec:
          nodeSelector:
            kubernetes.io/arch: amd64
          containers:
          - name: ray-head
            image: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_REPO}:${IMAGE_TAG}
            ports:
            - containerPort: 6379
              name: gcs
            - containerPort: 8265
              name: dashboard
            - containerPort: 8000
              name: serve
            resources:
              requests:
                cpu: "2"
                memory: "8Gi"
              limits:
                cpu: "2"
                memory: "8Gi"
            env:
            - name: PYTHONPATH
              value: /workspace
            volumeMounts:
            - name: serve-code
              mountPath: /workspace/vllm_serve.py
              subPath: vllm_serve.py
            readinessProbe:
              exec:
                command:
                - bash
                - -c
                - "ray health-check --address 127.0.0.1:6379 > /dev/null 2>&1"
              initialDelaySeconds: 10
              periodSeconds: 5
              timeoutSeconds: 3
              failureThreshold: 120
          volumes:
          - name: serve-code
            configMap:
              name: vllm-serve-script

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
            karpenter.sh/nodepool: gpu-nodepool
          initContainers:
          - name: fix-cache-perms
            image: busybox:1.36
            command: ["sh", "-c", "mkdir -p /cache/hub && chown -R 1000:1000 /cache"]
            volumeMounts:
            - name: cache-volume
              mountPath: /cache
          containers:
          - name: ray-worker
            image: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_REPO}:${IMAGE_TAG}
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
              value: /workspace
            - name: HF_HOME
              value: /mnt/k8s-disks/0/hf_cache
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: HF_TOKEN
            volumeMounts:
            - name: cache-volume
              mountPath: /mnt/k8s-disks/0/hf_cache
            - name: serve-code
              mountPath: /workspace/vllm_serve.py
              subPath: vllm_serve.py
            livenessProbe:
              exec:
                command:
                - bash
                - -c
                - "ray status > /dev/null 2>&1"
              initialDelaySeconds: 30
              periodSeconds: 5
              timeoutSeconds: 3
              failureThreshold: 120
            readinessProbe:
              exec:
                command:
                - bash
                - -c
                - "ray status > /dev/null 2>&1"
              initialDelaySeconds: 10
              periodSeconds: 5
              timeoutSeconds: 3
              failureThreshold: 10
          volumes:
          - name: cache-volume
            hostPath:
              path: /mnt/k8s-disks/0/hf_cache
              type: DirectoryOrCreate
          - name: serve-code
            configMap:
              name: vllm-serve-script
