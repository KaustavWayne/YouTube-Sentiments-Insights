import os
import json
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import dagshub

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
        logger.debug(f'Data loaded and NaNs filled from {file_path}')
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

def log_confusion_matrix(cm, file_name: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(file_name)
    plt.close()
    try:
        mlflow.log_artifact(file_name)
        logger.debug(f'Confusion matrix logged as artifact: {file_name}')
    except Exception as e:
        logger.warning(f"Skipping MLflow logging for confusion matrix: {e}")

def save_model_info(run_id: str, model_path: str, file_path: str):
    model_info = {
        'run_id': run_id,
        'model_path': model_path
    }
    with open(file_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    logger.debug(f'Model info saved to {file_path}')

# -------------------------
# Main pipeline
# -------------------------
def main():
    # Initialize MLflow with DagsHub
    dagshub.init(repo_owner='KaustavWayne', repo_name='YouTube-Sentiments-Insights', mlflow=True)
    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run() as run:
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

            # Load parameters
            params = load_params(os.path.join(root_dir, 'params.yaml'))
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Log model parameters
            if hasattr(model, 'get_params'):
                for param_name, param_value in model.get_params().items():
                    mlflow.log_param(param_name, param_value)

            # -------------------------------
            # Save experiment_info.json first
            # -------------------------------
            experiment_info_path = os.path.join(root_dir, 'experiment_info.json')
            save_model_info(run.info.run_id, "lgbm_model", experiment_info_path)

            # -------------------------------
            # Log model folder artifact
            # -------------------------------
            try:
                mlflow.sklearn.log_model(model, artifact_path='lgbm_model', registered_model_name='YouTube_Sentiment_LGBM')
            except Exception as e:
                logger.warning(f"MLflow model logging failed (DagsHub limitation): {e}")
                try:
                    mlflow.log_artifact(os.path.join(root_dir, 'lgbm_model.pkl'))
                except Exception as fallback_error:
                    logger.error(f"Failed to log model as artifact: {fallback_error}")

            # Log vectorizer
            try:
                mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
            except Exception as e:
                logger.warning(f"Vectorizer artifact logging failed: {e}")

            # Load test data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            X_test = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Evaluate model
            report, cm = evaluate_model(model, X_test, y_test)

            # Log metrics
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, os.path.join(root_dir, 'confusion_matrix_Test Data.png'))

            # Log classification report
            try:
                report_file = os.path.join(root_dir, 'classification_report.json')
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                mlflow.log_artifact(report_file)
            except Exception as e:
                logger.warning(f"Could not log classification report: {e}")

            # Set tags
            mlflow.set_tag('model_type', 'LightGBM')
            mlflow.set_tag('task', 'Sentiment Analysis')
            mlflow.set_tag('dataset', 'YouTube Comments')

            logger.info("Model evaluation completed successfully with organized artifacts")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
