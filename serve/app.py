import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from transformers import AutoTokenizer

fastapi_app = FastAPI()

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-2B")


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
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    @fastapi_app.get("/health")
    async def health(self):
        return {"status": "ok"}

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(self, request: dict):
        messages = request.get("messages", [])

        # Use tokenizer's chat template for proper prompt formatting
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=float(request.get("temperature", 0.7)),
            max_tokens=int(request.get("max_tokens", 256)),
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
