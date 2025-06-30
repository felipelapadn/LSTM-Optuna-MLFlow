import os
import mlflow
import math
import optuna
from dotenv import load_dotenv
load_dotenv()

mlflow.set_experiment(os.getenv("EXPERIMENT_NAME"))

class TrainModel:

    def __init__(self, model):
        self.model = model
        
    def champion_callback(self, study, frozen_trial):
        """
        Logging callback that will report when a new trial iteration improves upon existing
        best trial values.

        Note: This callback is not intended for use in distributed computing systems such as Spark
        or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
        workers or agents.
        The race conditions with file system state management for distributed trials will render
        inconsistent values with this callback.
        
        ref: https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs#configure-the-tracking-server-uri
        """

        winner = study.user_attrs.get("winner", None)

        if study.best_value and winner != study.best_value:
            study.set_user_attr("winner", study.best_value)
            if winner:
                improvement_percent = (
                    abs(winner - study.best_value) / study.best_value) * 100
                print(
                    f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                    f"{improvement_percent: .4f}% improvement"
                )
            else:
                print(
                    f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
    
    def run(self, run_name):
        """
        Executa a busca de hiperparâmetros para o modelo LSTM usando Optuna.

        Esta função instancia um estudo do Optuna para realizar a otimização dos
        hiperparâmetros do modelo LSTM. A cada iteração, as métricas de desempenho
        são registradas no MLflow para rastreamento e análise dos experimentos.
        """
        with mlflow.start_run(run_name=run_name, nested=True):
            study = optuna.create_study(direction="minimize")
            study.optimize(self.model.objective, n_trials=5, callbacks=[self.champion_callback])

            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_mse", study.best_value)
            mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

            mlflow.set_tags(
                tags={
                    "project": "Medical Cost Personal Datasets",
                    "optimizer_engine": "optuna",
                    "model_family": "LSTM",
                    "feature_set_version": 1,
                }
            )
