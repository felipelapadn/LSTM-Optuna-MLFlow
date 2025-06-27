import mlflow
from train_model import TrainModel
from lstm_model import LSTMModelOptimization
from utils import compare_and_register_model, make_train_test

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_id=371857808035113852)

    path = "./data/processed/processed_insurance.csv"
    target = "charges_scaled"

    X_train, X_test, y_train, y_test = make_train_test(path, target)
    lstm = LSTMModelOptimization(X_train, X_test, y_train, y_test )
    train_model = TrainModel(lstm)
    train_model.run()
    compare_and_register_model(X_train, X_test, y_train, y_test)
    
if __name__=="__main__":
    main()