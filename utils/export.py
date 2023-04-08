import argparse
import os
import pandas as pd
import wandb
import numpy as np

filter_keywords = {"do_shape": ["True", "False"]}
metric = "eval/mean_reward"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="bs-rl")
    parser.add_argument("--entity", type=str, default="qcoolers")
    parser.add_argument("--dir", type=str, default="export")
    parser.add_argument("-s", "--sweep_id", type=str, default="deuayze5")
    args = parser.parse_args()
    api = wandb.Api()
    entity, project, sweep_id, direc = args.entity, args.project, args.sweep_id, args.dir

    if not os.path.exists(direc):
        os.mkdir(direc)

    runs = api.runs(path=f"{entity}/{project}")
    
    for run in runs:
        if all([filter_kword in run.config for filter_kword in filter_keywords.keys()]):
            filter_dict = {k: run.config[k] for k in filter_keywords}
            run_df = pd.DataFrame(run.history())

            for k, v in filter_dict.items():
                run_df[k] = np.repeat(v, len(run_df))
            
            # Grab only the relevant columns
            columns = ["global_step", metric] + list(filter_keywords.keys())
            run_df = run_df[columns]
            # remove learning starts
            run_df = run_df[run_df[metric] != None]
            # Filter out "NaN"'s:
            run_df = run_df[run_df[metric].notna()]


            # A filename with the filter_dict info:
            file_name = "-".join([f"{k}-{v}" for k, v in filter_dict.items()])
            run_df.to_csv(f"{direc}/{run.id}-{file_name}.csv")

    print("Total runs in sweep:", len(runs))
    print("Total runs exported:", len(os.listdir(direc)))
    for key, values in filter_keywords.items():
        for value in values:
            print(f"Number of files with {key}={value}: " + \
                f"{len([f for f in os.listdir(direc) if f'{key}-{value}' in f])}")