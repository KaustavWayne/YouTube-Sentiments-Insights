
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
mlflow.set_tracking_uri("https://dagshub.com/KaustavWayne/YouTube-Sentiments-Insights.mlflow")
dagshub.init(repo_owner='KaustavWayne', repo_name='YouTube-Sentiments-Insights', mlflow=True)