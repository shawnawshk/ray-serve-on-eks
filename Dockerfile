FROM rayproject/ray:2.54.0-py312-gpu

# Install vllm (latest, currently v0.17.x) — compatible with Ray 2.54
# scipy upgrade needed: Ray 2.54 ships scipy 1.11 which uses numpy.Inf (removed in numpy 2.x)
RUN pip install --no-cache-dir vllm prometheus_client "scipy>=1.14.0"

# Application code is NOT baked in — it's mounted via ConfigMap at runtime.
# Set PYTHONPATH so Ray Serve can find the mounted script.
ENV PYTHONPATH=/workspace
