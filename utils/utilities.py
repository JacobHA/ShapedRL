import wandb

def test_sweep_existence(full_sweep_id):
    api = wandb.Api()
    try:
        api.sweep(full_sweep_id)
    except wandb.errors.CommError:
        print(f"Could not find sweep {full_sweep_id}.\n" +\
              "Double check that the project and sweep id exist.\n" +\
              "For a new sweep, run sweep.py without the -s flag.\nExiting...")
        exit(1)