import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


def train_model(X_train, y_train, X_test, y_test, preprocessor):

    # ✅ FIX ADDED HERE
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit-default-experiment")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }

    best_model = None
    best_f1 = 0

    for name, model in models.items():

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        with mlflow.start_run(run_name=name):

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(pipeline, "model")

            if f1 > best_f1:
                best_f1 = f1
                best_model = pipeline

    joblib.dump(best_model, "models/best_model.pkl")

    return best_model
