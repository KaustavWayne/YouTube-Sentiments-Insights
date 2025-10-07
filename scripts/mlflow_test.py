import dagshub
import mlflow
import random

# Initialize DagsHub MLflow integration
dagshub.init(repo_owner='KaustavWayne', repo_name='YouTube-Sentiments-Insights', mlflow=True)

# Start an MLflow run
with mlflow.start_run():
    # Log random parameters
    mlflow.log_param("param1", random.randint(1, 100))
    mlflow.log_param("param2", random.random())

    # Log random metrics
    mlflow.log_metric("metric1", random.random())
    mlflow.log_metric("metric2", random.uniform(0.5, 1.5))

    print("✅ Logged random parameters and metrics to DagsHub MLflow.")
