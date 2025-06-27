import mlflow
import pandas as pd
from lstm_model import LSTMModelOptimization
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

client = MlflowClient()
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id=371857808035113852)

def make_train_test(path, target):

    if path.endswith(".csv"):
        data = pd.read_csv(path)
    elif path.endswith(".parquet"):
        data = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
    
    X = data.drop([target], axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def compare_and_register_model(X_train, X_test, y_train, y_test, n_runs=5):
    runs = client.search_runs(
        experiment_ids=[371857808035113852],                  
        max_results=n_runs
    )

    model_lstm_obj = LSTMModelOptimization(X_train, X_test, y_train, y_test)
    model_name = "exp-lstm-test"
    model_version = "latest"

    versions = client.get_latest_versions(model_name)
    if versions == []:
        model_trained, score = model_lstm_obj.train_model(runs[0].data.params)
        mlflow.keras.log_model(model_trained, "model", registered_model_name=model_name)
    else:
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.keras.load_model(model_uri)
        latest_score = model.evaluate(X_test, y_test, verbose=0)
        model_to_att = None
        for run in runs:
            model_trained, score = model_lstm_obj.train_model(run.data.params)
            print(f"Score: {score[0]} | Latest Score: {latest_score[0]}")
            if score[0] < latest_score[0]:
                model_to_att = model_trained

        if model_to_att is not None:
            mlflow.keras.log_model(model_to_att, "model", registered_model_name=model_name)