### Run experiments based on globally set variables

```

Specify the experiment parameters in the config.py file. 
The wandb config files for each algo are in the sweep_configs folder.

```
python experiment.py --sweep_id="mysweepid" --count=10
```

# TODOs:
- [x] Add a ShapedTD3 class
- [x] Automatically choose the sweep config.yml file based on model used