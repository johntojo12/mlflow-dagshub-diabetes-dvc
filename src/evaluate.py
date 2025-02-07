import pandas as pd
import pickle 
from sklearn.metrics import accuracy_score
import yaml 
import os
import json  # Importing json to save the results as a JSON file
import mlflow

# Set MLflow tracking URI and credentials
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/johntojo12/my-first-repo.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'johntojo12'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'dc292bac8c583ddb64c7eed6194c014db8d04eae'

# Load the params.yaml file
params = yaml.safe_load(open('params.yaml'))['train']

def evaluate(data_path, model_path, output_path):
    # Load data and model
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    # Load the model
    model = pickle.load(open(model_path, 'rb'))
    
    # Make predictions
    prediction = model.predict(X)
    accuracy = accuracy_score(y, prediction)
    
    # Log the metrics to MLflow
    mlflow.log_metric('accuracy', accuracy)
    print("Model accuracy:", accuracy)
    
    # Save the evaluation result to a JSON file
    evaluation_result = {
        'accuracy': accuracy
    }
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the evaluation results to a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(evaluation_result, json_file)
    
    print(f"Evaluation report saved to {output_path}")
    
if __name__ == '__main__':
    # Ensure that the output path is passed along
    evaluate(params['data'], params['model'], 'results/evaluation_report.json')
