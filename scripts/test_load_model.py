import os

import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

# --- DagsHub MLflow setup ---
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# Use token for authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "KaustavWayne"
repo_name = "YouTube-Sentiments-Insights"

# Set MLflow tracking URI to the DagsHub repo
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# --- Test for loading latest model from DagsHub ---
@pytest.mark.parametrize("model_name, stage", [
    ("yt_chrome_plugin_model", "staging"),
])
def test_load_latest_staging_model(model_name, stage):
    client = MlflowClient()
    
    # Get latest version in the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in '{stage}' stage for '{model_name}'"

    try:
        # Load latest model version
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version} loaded successfully from '{stage}' stage on DagsHub.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
