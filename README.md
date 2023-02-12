### Run experiments based on globally set variables

Specify the experiment parameters in the config.py file. (No arguments are passed to the experiment.py file)
The wandb config files for each algo are in the sweep_configs folder.

```
python experiment.py
```

# TODOs:
- [x] Add a ShapedTD3 class
- [x] Automatically choose the sweep config.yml file based on model used
- [ ] Clean up sweep id configuration

# Related Work:
- https://arxiv.org/pdf/2011.02669.pdf

