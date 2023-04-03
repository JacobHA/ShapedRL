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
- [ ] Clean up eval callback 

# Related Work:
- https://arxiv.org/pdf/2011.02669.pdf
- Munchausen: https://arxiv.org/abs/2007.14430
- (related to above: https://arxiv.org/abs/2205.07467)

# Future Work / Ideas:
- For automatically adjusting the learning rate (to min. clipping) or adjust the soft-clip weight parameter:
- Sec 5 in https://arxiv.org/pdf/1812.05905.pdf
