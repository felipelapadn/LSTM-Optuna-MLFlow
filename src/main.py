import os
import mlflow
from dotenv import load_dotenv

from train_model import TrainModel
from lstm_model import LSTMModelOptimization
from utils import compare_and_register_model, make_train_test

load_dotenv()

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_id = os.getenv("EXPERIMENT_ID")
    run_name = os.getenv("RUN_NAME")
    model_name = os.getenv("MODEL_NAME")
    dataset_path = os.getenv("DATA_PATH")
    target_column = os.getenv("TARGET_COLUMN")

    if not all([tracking_uri, experiment_id, run_name, dataset_path, target_column]):
        raise ValueError("Verifique se todas as variáveis de ambiente (.env) estão definidas.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=experiment_id)

    X_train, X_test, y_train, y_test = make_train_test(dataset_path, target_column)
    lstm_optimizer = LSTMModelOptimization(X_train, X_test, y_train, y_test)
    model_trainer = TrainModel(lstm_optimizer)

    model_trainer.run(run_name)
    compare_and_register_model(experiment_id, model_name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
