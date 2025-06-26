import mlflow
import pandas as pd
from train_model import TrainModel
from lstm_model import LSTMModelOptimization
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from mlflow.exceptions import RestException


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id=371857808035113852)


def make_train_test(self):

    if self.path.endswith(".csv"):
        data = pd.read_csv(self.path)
    elif self.path.endswith(".parquet"):
        data = pd.read_parquet(self.path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
    
    X = data.drop([self.target], axis=1)
    y = data[self.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def register_model(model, model_name):
    model_version = "latest"
    model_uri = f"models:/{model_name}/{model_version}"
    try:
        mlflow.keras.load_model(model_uri)
    except RestException:
        print(f"[INFO] Modelo '{model_name}' ainda não registrado. Criando a primeira versão.")
        mlflow.keras.log_model(model, "model", registered_model_name=model_name)
        return

X, y = make_regression(n_samples=1000, n_features=4, noise=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# lstm = LSTMModelOptimization(X_train, X_test, y_train, y_test )
# train_model = TrainModel(lstm)
# train_model.run()

from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment_id = "0"  

runs = client.search_runs(
    experiment_ids=[371857808035113852],  
    filter_string="",                 
    max_results=5  
)

model_lstm_obj = LSTMModelOptimization(X_train, X_test, y_train, y_test)
for run in runs:
    
    model_trained, score = model_lstm_obj.train_model(run.data.params)
    compare_and_register_model_(model_trained, "exp-lstm-test")
    
    model_name = "exp-lstm-test"
    model_version = "latest"

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.keras.load_model(model_uri)
    lates_score = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Score: {score} | Latest Score: {lates_score}")
    


# model_trained

# mlflow.sklearn.log_model(
#     sk_model=model_trained,
#     name="sklearn-model-sgd",
#     input_example=X,
#     registered_model_name="sk-learn-sgd-reg-model",
# )