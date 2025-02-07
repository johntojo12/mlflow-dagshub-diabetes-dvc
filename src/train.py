import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
import mlflow

# Set MLflow tracking URI and credentials
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/johntojo12/my-first-repo.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'johntojo12'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'dc292bac8c583ddb64c7eed6194c014db8d04eae'

def hyperparameter_tuning(X_train, y_train, param_grid):
    """Perform hyperparameter tuning using GridSearchCV."""
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

def train(data_path, model_path, random_state, param_grid):
    """Train the model and log results using MLflow."""
    # Load data
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    # Start MLflow run
    with mlflow.start_run():
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        
        # Infer signature based on training data
        signature = infer_signature(X_train, y_train)

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        # Evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Log metrics and parameters
        mlflow.log_metric("accuracy", accuracy)
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)

        # Log confusion matrix and classification report as artifacts
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        # Log confusion matrix as an artifact
        with open("confusion_matrix.txt", "w") as cm_file:
            cm_file.write(str(cm))
        mlflow.log_artifact("confusion_matrix.txt")
        
        # Log classification report as an artifact
        with open("classification_report.txt", "w") as cr_file:
            cr_file.write(cr)
        mlflow.log_artifact("classification_report.txt")

        # Log the model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != 'file':
            # Log model with input example and signature for remote tracking
            mlflow.sklearn.log_model(best_model, 'model', registered_model_name='Best Model', signature=signature, input_example=X_train.iloc[0].to_dict())
            print("Model logged as HTTP.")
        else:
            # Log model as file for local storage
            mlflow.sklearn.log_model(best_model, 'model', signature=signature)
            print("Model logged as file.")

        # Save the model locally using pickle
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Model saved to {model_path}")

if __name__ == '__main__':
    # Load parameters from YAML file
    params = yaml.safe_load(open('params.yaml'))['train']
    data_path = params['data']
    model_path = params['model']
    random_state = params['random_state']
    param_grid = {
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf']
    }
    
    train(data_path, model_path, random_state, param_grid)
