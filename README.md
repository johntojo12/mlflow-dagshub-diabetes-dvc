 project: data pipeline with dvc abd mlflow for ML


Notes
dvc init 
dvc add data/data.csv #to track the file using dvc

track md5 using git present in data.csv.dvs and also gitignore (git add .\data\.gitignore .\data\data.csv.dvc)

In ndagshub remote-- in data--dvc--dagshub_dvc_remote--need to config
https://dagshub.com/johntojo12/my-first-repo
(repo)
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/johntojo12/my-first-repo.s3
need to setup credentials
dvc remote modify origin --local access_key_id (expand eye icon)
dvc remote modify origin --local secret_access_key_(expand eye icon)
dvc remote list
dvc pull -r origin
dvc push -r origin #wont be reflected in dagshub as wen need to do gitpush
git push origin main

u made change in data.csv
do the following 
dvc add .\data\data.csv the md5 value gets updated
git add .
git commit -m '?'
dvc pull - r origin
dvc push -r origin
git push origin main



#project starts from here 
start with params.yaml
then worked on preprocess.py

before working on train.py 
need to config dagshub --remote--expirements
3 var set
 1 with mlflow
  os.environ['MLFLOW_TRACKING_URI']= https://dagshub.com/johntojo12/my-first-repo.mlflow 
  2 user 
  os.environ['MLFLOW_TRACKING_USERNAME']='johntojo12'
  3 PASSWORD get it from dvc dagshub setup credientals secret access key
  os.environ['MLFLOW_TRACKING_PASSWORD']=''

  develope the train pipeline and then move to evaluate pipeline

  connect all 3 pipelines and integrate with dvc

  dvc init 
  want to track the raw data
  dvc add data/raw/data.csv
  git add data\raw\data.csv.dvc
  git add data\raw\.gitignore

  combining all 3 pipelines
  dvc stage is used define the workflow how they have to be executed and can be tracked in dvc.yaml file
-n name of the process
-p tracks the parameter
-d dependencies for the stage
-o output of the stage

creating stage 1
run in gitbash 
source actiavte env_name
  dvc stage add -n preprocess \
    -p preprocess.input, preprocess.output \
    -d src/preprocess.py -d dara/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py
or in power shell
dvc stage add -n preprocess -p preprocess.input,preprocess.output -d src/preprocess.py -d data/raw/data.csv -o data/processed/data.csv python src/preprocess.py

creating stage 2
dvc stage add -n train \
    -p 
    train.data,train.model,train.random_state,train.n_estimators,train.max_depth,train.min_samples_split,train.min_samples_leaf \
    -d src/train.py -d data/processed/data.csv \
    -o models/model.pkl
    python src/train.py

    dvc stage add -n train `
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth,train.min_samples_split,train.min_samples_leaf `
    -d src/train.py -d data/processed/data.csv `
    -o models/model.pkl `
    python src/train.py

stage 3
dvc stage add -n evaluate  \
 -d src/evaluate.py -d models/model.pkl -d data/processed.data/csv \
 python src/evaluate.py

 dvc stage add -n evaluate `
    -d src/evaluate.py -d models/model.pkl -d data/processed/data.csv `
    -o results/evaluation_report.json `
    python src/evaluate.py

to run the entire pipeline
dvc repro

need to setup dagshub dvc remote
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/johntojo12/my-first-repo.s3
setup the credentials
dvc remote modify origin --local access_key_id dc292bac8c583ddb64c7eed6194c014db8d04eae
dvc remote modify origin --local secret_access_key dc292bac8c583ddb64c7eed6194c014db8d04eae

dvc pull -r origin
dvc push -r origin
git add .
git commit -m'final changes'
git push origin 
