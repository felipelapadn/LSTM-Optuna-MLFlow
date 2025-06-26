import mlflow
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id=371857808035113852)
# mlflow.set_registry_uri("model-test")

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([1, 1, 2, 2])
y_test = np.array([0])
    
with mlflow.start_run() as run:

    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    clf.fit(X, Y)
    pred = clf.predict([[-0.8, -1]])
    
    acc_train = accuracy_score(y_test, pred)

    mlflow.log_metrics({"acc": acc_train})
    
    # deixar o autolog, mas colocar um if para se a acc ser melhor daí sim faz o load
    
    mlflow.sklearn.log_model(
        sk_model=clf,
        name="sklearn-model-sgd",
        input_example=X,
        registered_model_name="sk-learn-sgd-reg-model",
    )

# model_uri = f"runs:/{run.info.run_id}/sklearn-model-sgd"
# mv = mlflow.register_model(model_uri, "sk-learn-sgd-reg-model")
# print(f"Name: {mv.name}")
# print(f"Version: {mv.version}")

    
    