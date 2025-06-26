import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id=371857808035113852)

model_name = "sk-learn-sgd-reg-model"
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

print(model)