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
