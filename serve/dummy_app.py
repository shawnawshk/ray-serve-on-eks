# Dummy app for local testing (no GPU/vLLM required)
# Validates: Ray Serve deployment, ConfigMap mount, PYTHONPATH, health endpoint
from fastapi import FastAPI
from ray import serve

fastapi_app = FastAPI()


@serve.deployment(
    ray_actor_options={"num_cpus": 0.5},
    max_ongoing_requests=10,
)
@serve.ingress(fastapi_app)
class DummyDeployment:
    @fastapi_app.get("/health")
    async def health(self):
        return {"status": "ok"}

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(self, request: dict):
        messages = request.get("messages", [])
        prompt = messages[-1]["content"] if messages else "empty"
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "dummy",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"Echo: {prompt}"},
                    "finish_reason": "stop",
                }
            ],
        }


deployment = DummyDeployment.bind()
