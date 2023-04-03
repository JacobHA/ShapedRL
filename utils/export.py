import argparse
import os
import pandas as pd
import wandb
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="bs-rl")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--dir", type=str, default="export")
    args = parser.parse_args()
    api = wandb.Api()
    entity, project, dir = args.entity, args.project, args.dir

    if not os.path.exists(dir):
        os.mkdir(dir)

    runs = api.runs(entity + "/" + project)

    for run in runs:
        if "map_name" in run.config and "clip_method" in run.config:
            map_name = run.config["map_name"]
            clip_method = run.config["clip_method"]
            run_df = pd.DataFrame(run.history())
            run_df["clip_method"] = np.repeat(clip_method, len(run_df))
            run_df.to_csv(f"{dir}/{map_name}-{clip_method}-{run.name}.csv")