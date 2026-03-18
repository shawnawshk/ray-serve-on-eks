include .env
export

IMAGE_URI := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(IMAGE_REPO)

.PHONY: help build push login configmap render deploy deploy-webui destroy load-test all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

login: ## Login to ECR
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

build: ## Build the runtime image
	docker buildx build --platform linux/amd64 -t $(IMAGE_URI):$(IMAGE_TAG) --load .

push: login ## Push the runtime image to ECR
	docker push $(IMAGE_URI):$(IMAGE_TAG)

build-push: login ## Build and push in one step
	docker buildx build --platform linux/amd64 -t $(IMAGE_URI):$(IMAGE_TAG) --push .

configmap: ## Generate ConfigMap from serve/vllm_serve.py and apply
	kubectl create configmap vllm-serve-script \
		--from-file=vllm_serve.py=serve/vllm_serve.py \
		--dry-run=client -o yaml | kubectl apply -f -

render: ## Render rayservice.yaml from template with .env values
	@IMAGE_URI=$(IMAGE_URI) IMAGE_TAG=$(IMAGE_TAG) envsubst < k8s/ray/rayservice.yaml.tpl > k8s/ray/rayservice.yaml
	@echo "Rendered k8s/ray/rayservice.yaml with image $(IMAGE_URI):$(IMAGE_TAG)"

deploy: configmap render ## Deploy ConfigMap + RayService
	kubectl apply -f k8s/ray/rayservice.yaml

deploy-webui: ## Deploy Open WebUI
	kubectl apply -f k8s/ray/open-webui.yaml

destroy: ## Delete RayService + ConfigMap + Open WebUI
	-kubectl delete rayservice vllm-serve -n default
	-kubectl delete configmap vllm-serve-script -n default
	-kubectl delete -f k8s/ray/open-webui.yaml

load-test: ## Run Locust load test headless (HOST, USERS, SPAWN_RATE, RUN_TIME overrideable)
	@kubectl port-forward svc/vllm-serve-serve-svc 8000:8000 & \
	PF_PID=$$!; \
	trap "kill $$PF_PID 2>/dev/null" EXIT INT TERM; \
	sleep 2; \
	locust -f load-test/locustfile.py \
		--host $${HOST:-http://localhost:8000} \
		--headless \
		-u $${USERS:-8} \
		-r $${SPAWN_RATE:-2} \
		--run-time $${RUN_TIME:-10m}

all: build-push deploy ## Build, push, and deploy everything
