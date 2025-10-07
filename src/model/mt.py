import os
import json
import pickle
import logging
import yaml
import mlflow
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import dagshub
import shutil

# -------------------------
# Logging configuration
# -------------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -------------------------
# Helper functions
# -------------------------
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug(f'Data loaded from {file_path}')
        return df
    except Exception as e:
        logger.error(f'Error loading data from {file_path}: {e}')
        raise

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f'Model loaded from {model_path}')
        return model
    except Exception as e:
        logger.error(f'Error loading model from {model_path}: {e}')
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.debug(f'Vectorizer loaded from {vectorizer_path}')
        return vectorizer
    except Exception as e:
        logger.error(f'Error loading vectorizer from {vectorizer_path}: {e}')
        raise

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug(f'Parameters loaded from {params_path}')
        return params
    except Exception as e:
        logger.error(f'Error loading parameters from {params_path}: {e}')
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation completed')
        return report, cm
    except Exception as e:
        logger.error(f'Error during model evaluation: {e}')
        raise

def log_confusion_matrix(cm, file_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(file_path)
    plt.close()
    logger.debug(f'Confusion matrix saved to {file_path}')

def save_artifacts_folder(root_dir, model, vectorizer, report, cm):
    artifacts_dir = os.path.join(root_dir, 'Artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save model in Artifacts/model/
    model_dir = os.path.join(artifacts_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # Optional: save MLflow-style metadata files
    with open(os.path.join(model_dir, 'MLmodel'), 'w') as f:
        f.write("artifact: model.pkl\n")
    with open(os.path.join(model_dir, 'conda.yaml'), 'w') as f:
        f.write("name: model-env\ndependencies: []\n")
    with open(os.path.join(model_dir, 'python_env.yaml'), 'w') as f:
        f.write("name: python-env\ndependencies: []\n")
    with open(os.path.join(model_dir, 'requirements.txt'), 'w') as f:
        f.write("")

    # Save vectorizer
    with open(os.path.join(artifacts_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save metrics
    metrics_file = os.path.join(artifacts_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Save confusion matrix
    cm_file = os.path.join(artifacts_dir, 'confusion_matrix_Test Data.png')
    log_confusion_matrix(cm, cm_file)

    # Log everything to MLflow
    mlflow.log_artifacts(artifacts_dir)
    logger.debug(f'All artifacts saved and logged in {artifacts_dir}')


# -------------------------
# Main pipeline
# -------------------------
def main():
    mlflow.set_tracking_uri('https://dagshub.com/KaustavWayne/YouTube-Sentiments-Insights.mlflow')
    dagshub.init(repo_owner='KaustavWayne', repo_name='YouTube-Sentiments-Insights', mlflow=True)
    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run():
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Log model parameters
            if hasattr(model, 'get_params'):
                for param_name, param_value in model.get_params().items():
                    mlflow.log_param(param_name, param_value)

            # Load test data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            X_test = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Evaluate model
            report, cm = evaluate_model(model, X_test, y_test)

            # Add tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

            # Save everything in organized Artifacts folder
            save_artifacts_folder(root_dir, model, vectorizer, report, cm)

            logger.info("Model evaluation completed successfully.")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
