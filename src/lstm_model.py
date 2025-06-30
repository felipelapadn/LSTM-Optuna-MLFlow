import os
import keras
import mlflow
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

class LSTMModelOptimization:
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=30, restore_best_weights=True)

    def create_model(self, params):
        """
        Cria uma instância do modelo LSTM com base nos parâmetros fornecidos.

        Args:
            params (dict): Dicionário contendo os hiperparâmetros para instanciar o modelo,
                        geralmente sugeridos por um trial do Optuna.

        Returns:
            keras.models.Sequential: Modelo LSTM do Keras construído com as camadas definidas.
        """
        
        required_params = [
            "input_shape", "units", "dropout", "n_layers", 
            "optimizer", "learning_rate", "batch_size"
        ]

        missing = [p for p in required_params if p not in params]
        if missing:
            raise ValueError(f"Parâmetros ausentes: {missing}")
        else:
            model = Sequential()
            model.add(LSTM(units=int(params["units"]), return_sequences=True, input_shape=(
                int(params["input_shape"]), 1)))
            model.add(Dropout(float(params["dropout"])))

            for _ in range(int(params["n_layers"]) - 1):
                model.add(LSTM(units=int(params["units"]), return_sequences=True))
                model.add(Dropout(float(params["dropout"])))

            model.add(LSTM(units=int(params["units"])))
            model.add(Dropout(float(params["dropout"])))
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer=params["optimizer"],
                        loss='mean_squared_error', metrics=['mse'])

            return model

    def objective(self, trial):
        """
        Função objetivo utilizada pelo Optuna para otimização do modelo LSTM.

        Durante cada trial, o modelo é treinado com os hiperparâmetros sugeridos e suas
        métricas de desempenho são registradas no MLflow.

        Args:
            trial (optuna.trial.Trial): Objeto trial fornecido pelo Optuna, usado para sugerir os hiperparâmetros.

        Returns:
            float: Métrica de erro quadrático médio (MSE) do modelo no conjunto de validação.
        """
        with mlflow.start_run(experiment_id=os.getenv("EXPERIMENT_ID"),
                              run_name=f"run-optuna-exp-{trial.number}", nested=True):
            params = {
                "input_shape": self.x_train.shape[1],
                "units": trial.suggest_int('units', 32, 256, step=32),
                "dropout": trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
                "n_layers": trial.suggest_int('n_layers', 1, 5), 
                "optimizer": trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'nadam']),
                "learning_rate": trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),  
                "batch_size": trial.suggest_categorical('batch_size', [32, 64, 128]), # aqui é legal usar o suggest_categorical
            }                                                                           # para ser um valor sem variação dentre esses da lista

            model = self.create_model(params)
            model.fit(
                self.x_train, self.y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=int(params["batch_size"]),
                callbacks=[self.callback],
                verbose=0
            )

            score = model.evaluate(self.x_test, self.y_test, verbose=0)

            mlflow.log_params(params)
            mlflow.log_metric("mse", score[1])
            mlflow.log_metric("rmse", math.sqrt(score[1]))

        return score[1]
    
    def train_model(self, params):
        """
        Função responsavel por fazer o treinamento do modelo de maneira direta, sem o Optuna.

        Args:
            params (dict): Dicionário contendo os hiperparâmetros para instanciar o modelo,
                        geralmente sugeridos por um trial do Optuna.

        Returns:
            keras.models.Sequential: Modelo LSTM do Keras construído com as camadas definidas.
            float: Métrica de erro quadrático médio (MSE) do modelo no conjunto de validação.
        """
        model = self.create_model(params)

        model.fit(
            self.x_train, self.y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=int(params["batch_size"]),
            callbacks=[self.callback],
            verbose=0
        )
        
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        return model, score
        

