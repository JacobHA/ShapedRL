Hello, reviewer! Thank you for taking the time to read our submission and visit the supplementary code. With this repository, you can re-run the experiments from the paper. H

We would like to emphasize that the BSRS can succinctly be summarized as changing one line of code in the Bellman target equation:

```
rewards + gamma * next_state_values
```
to 
```
rewards + gamma * eta * next_state_values - eta * state_values + gamma * eta * next_state_values
```

### Run experiments based on globally set variables

Specify the experiment parameters in the config.py file. (No arguments are passed to the experiment.py file)
The wandb config files for each algo are in the sweep_configs folder.

```
python experiment.py
```
