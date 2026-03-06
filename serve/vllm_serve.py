"""
Ray Serve + vLLM OpenAI-compatible serving deployment.

Wraps vLLM's native OpenAI serving layer (chat completions, completions, models)
behind a Ray Serve deployment. Application code is decoupled from the container
image and mounted via ConfigMap.

Tested with vLLM v0.17.x + Ray 2.54.x.
"""

import os
import re
from typing import Optional

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Mount
from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.logger import init_logger

logger = init_logger(__name__)

app = FastAPI()


@serve.deployment(
    name="vllm-deployment",
    ray_actor_options={"num_gpus": 1},
    health_check_period_s=10,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 32,
        max_model_len: int = 4096,
        enforce_eager: bool = True,
        gpu_memory_utilization: float = 0.9,
        response_role: str = "assistant",
        chat_template: Optional[str] = None,
    ):
        # Prometheus metrics endpoint
        metrics_app = make_asgi_app()
        route = Mount("/metrics", metrics_app)
        route.path_regex = re.compile("^/metrics(?P<path>.*)")
        app.routes.append(route)

        logger.info(f"Initializing VLLMDeployment with model={model}")
        self.model_path = model

        engine_args = AsyncEngineArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            dtype="auto",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_chunked_prefill=True,
        )

        try:
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise

        self.response_role = response_role
        self.chat_template = chat_template
        self.openai_serving_chat = None
        self.openai_serving_completion = None
        self.serving_models = None

    async def _init_serving_components(self):
        """Lazy-init OpenAI serving components after engine is fully ready."""
        if self.serving_models is not None:
            return

        base_model_paths = [
            BaseModelPath(name=self.model_path, model_path=self.model_path)
        ]

        self.serving_models = OpenAIServingModels(
            engine_client=self.engine,
            base_model_paths=base_model_paths,
        )

        self.openai_serving_completion = OpenAIServingCompletion(
            engine_client=self.engine,
            models=self.serving_models,
            request_logger=None,
        )

        self.openai_serving_chat = OpenAIServingChat(
            engine_client=self.engine,
            models=self.serving_models,
            response_role=self.response_role,
            request_logger=None,
            chat_template=self.chat_template,
            chat_template_content_format="auto",
            enable_auto_tools=os.environ.get(
                "ENABLE_AUTO_TOOL_CHOICE", "false"
            ).lower()
            in ("1", "true", "yes"),
            tool_parser=os.environ.get("TOOL_CALL_PARSER", "").strip() or None,
            enable_prompt_tokens_details=False,
        )

    async def health_check(self) -> str:
        """Ray Serve health check."""
        return "OK"

    @app.get("/health")
    async def health(self):
        return {"status": "ok"}

    @app.get("/v1/models")
    async def get_models(self):
        await self._init_serving_components()
        models_list = await self.serving_models.show_available_models()
        return JSONResponse(content=models_list.model_dump())

    @app.post("/v1/completions")
    async def create_completion(
        self, request: CompletionRequest, raw_request: Request
    ):
        await self._init_serving_components()
        generator = await self.openai_serving_completion.create_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(
                content=generator, media_type="text/event-stream"
            )
        return JSONResponse(content=generator.model_dump())

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        await self._init_serving_components()
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(
                content=generator, media_type="text/event-stream"
            )
        return JSONResponse(content=generator.model_dump())


# --- Bind deployment with env-var overrides ---
deployment = VLLMDeployment.bind(
    model=os.environ.get("MODEL_ID", "Qwen/Qwen3.5-2B"),
    tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
    max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "32")),
    max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
    enforce_eager=os.environ.get("ENFORCE_EAGER", "true").lower()
    in ("1", "true", "yes"),
    gpu_memory_utilization=float(
        os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")
    ),
)
