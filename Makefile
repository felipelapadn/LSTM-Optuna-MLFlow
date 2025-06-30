IMAGE_NAME = optuna-lstm
CONTAINER_NAME = optuna-lstm
REPOSITORY_NAME = optuna-lstm
PORT = 8080

ifneq (,$(wildcard .env))
    include .env
    export
endif

docker: 
	@echo "Construindo a imagem Docker..."
	@docker build \
	--no-cache \
	-f Dockerfile \
	-t $(IMAGE_NAME) .

run:
	@echo "Executando o container Docker..."
	@docker run -d \
		--name $(CONTAINER_NAME) \
		-p 5000:5000 \
		-v $(shell pwd)/mlruns:/mlflow/mlruns \
		-v $(shell pwd)/mlartifacts:/mlflow/mlartifacts \
		-e MLFLOW_ARTIFACT_ROOT=/mlflow/mlartifacts \
		$(IMAGE_NAME) \
		bash -c "mlflow server --host 0.0.0.0 --port 5000"

clean:
	@echo "Limpando imagens e contêineres Docker..."
	@docker stop $(CONTAINER_NAME) || true
	@docker rm $(CONTAINER_NAME) || true
	@docker rmi $(IMAGE_NAME) || true