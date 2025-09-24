### Run experiments based on globally set variables

Specify the experiment parameters in the config.py file. (No arguments are passed to the experiment.py file)
The wandb config files for each algo are in the sweep_configs folder.

```
python experiment.py
```

# TODOs:
- [x] Fix TD3 classes for sampling continuous actions to calculate V(s) and V(s')
- [x] Automatically choose the sweep config.yml file based on model used
- [x] Clean up sweep id configuration
- [x] Clean up eval callback 

# Related Work:
- https://arxiv.org/pdf/2011.02669.pdf
- Munchausen: https://arxiv.org/abs/2007.14430
- (related to above: https://arxiv.org/abs/2205.07467)

# Future Work / Ideas:
- (Learned) schedule for eta as training progresses
- eta(s) - state-dependent theory holds
- Improve the bounds
