import mlflow
import math
import optuna
import pandas as pd


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id=371857808035113852)

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
    
    def run(self):
        
        with mlflow.start_run(run_name="run-optuna-exp-lstm", nested=True):
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

            # # Get the logged model uri so that we can load it from the artifact store
            # model_uri = mlflow.get_artifact_uri(artifact_path)
