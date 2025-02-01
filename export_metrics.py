import mlflow
import pandas as pd
import json
from pathlib import Path


experiment_id = "213424715458696362"
parent_run_id = "419b52fbcc094e1988be22d64825e44d"

client = mlflow.tracking.MlflowClient()

def get_all_nested_runs(run_id, experiment_id):
    nested_runs = []
    
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{run_id}'"
    )
    
    for run in child_runs:
        nested_runs.append(run)
        nested_runs.extend(get_all_nested_runs(run.info.run_id, experiment_id))
    
    return nested_runs

def extract_run_data(run):
    data = {
        "run_id": run.info.run_id,
        "parent_run_id": run.data.tags.get("mlflow.parentRunId", None),
        "params": run.data.params,
        "metrics": run.data.metrics,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
    }
    return data

parent_run = mlflow.get_run(parent_run_id)
all_runs_data = [extract_run_data(parent_run)]

nested_runs = get_all_nested_runs(parent_run_id, experiment_id)
all_runs_data.extend([extract_run_data(run) for run in nested_runs])

df = pd.json_normalize(all_runs_data)

df.to_csv('mlflow_data/mlflow_all_runs_data.csv', index=False)
with open('mlflow_data/mlflow_all_runs_data.json', 'w') as f:
    json.dump(all_runs_data, f, indent=4)


# ---------------------------------Artifacts---------------------------------

ARTIFACTS_DIR = Path("mlflow_data/mlflow_artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

def download_artifacts(run_id):

    run_artifact_dir = ARTIFACTS_DIR / run_id
    run_artifact_dir.mkdir(exist_ok=True)
    
    artifact_list = client.list_artifacts(run_id)
    
    for artifact in artifact_list:
        artifact_path = artifact.path
        local_path = client.download_artifacts(run_id, artifact_path, dst_path=run_artifact_dir)

download_artifacts(parent_run_id)

for run in nested_runs:
    download_artifacts(run.info.run_id)