import mlflow
import pandas as pd
from lstm_model import LSTMModelOptimization
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

client = MlflowClient()

def make_train_test(path, target):
    """
    Realiza a divisão entre treino e teste a partir de um dataset localizado no caminho especificado.

    Args:
        path (str): Caminho da pasta onde o dataset está localizado.
        target (str): Nome da coluna alvo (target) presente no dataset.

    Raises:
        ValueError: Se o arquivo não for dos tipos suportados (.csv ou .parquet).

    Returns:
        tuple: Tupla contendo (X_train, X_test, y_train, y_test), os conjuntos de treino e teste.
    """

    if path.endswith(".csv"):
        data = pd.read_csv(path)
    elif path.endswith(".parquet"):
        data = pd.read_parquet(path)
    else:
        raise ValueError("Tipo de arquivo não suportado. Use .csv ou .parquet")
    
    X = data.drop([target], axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def compare_and_register_model(exp_id, model_name_, X_train, X_test, y_train, y_test, n_runs=5):
    """
    Compara os modelos treinados pelo Optuna nas últimas `n_runs` execuções com as
    métricas da versão mais recente do modelo salvo.

    Args:
        X_train (array-like): Conjunto de dados de treino (features).
        X_test (array-like): Conjunto de dados de teste (features).
        y_train (array-like): Rótulos correspondentes ao conjunto de treino.
        y_test (array-like): Rótulos correspondentes ao conjunto de teste.
        n_runs (int, optional): Número de execuções (trials) anteriores do Optuna a
        serem consideradas na comparação. Defaults to 5.

    Returns:
        dict: Dicionário contendo as métricas de comparação entre os modelos testados e o modelo salvo.
    """
    runs = client.search_runs(
        experiment_ids=[exp_id],                  
        max_results=n_runs
    )

    model_lstm_obj = LSTMModelOptimization(X_train, X_test, y_train, y_test)
    model_name = model_name_
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