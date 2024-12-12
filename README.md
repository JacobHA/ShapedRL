### Run experiments based on globally set variables

Specify the experiment parameters in the config.py file. (No arguments are passed to the experiment.py file)
The wandb config files for each algo are in the sweep_configs folder.

```
python experiment.py
```

# TODOs:
- [ ] Fix SAC/TD3 classes for sampling continuous actions to calculate V(s) and V(s')
- [x] Automatically choose the sweep config.yml file based on model used
- [ ] Clean up sweep id configuration
- [x] Clean up eval callback 

# Related Work:
- https://arxiv.org/pdf/2011.02669.pdf
- Munchausen: https://arxiv.org/abs/2007.14430
- (related to above: https://arxiv.org/abs/2205.07467)

# Future Work / Ideas:
- For automatically adjusting the learning rate (to min. clipping) or adjust the soft-clip weight parameter:
- Sec 5 in https://arxiv.org/pdf/1812.05905.pdf

# Experiment evaluation on a remote workstation:
Steps for reproducing the experiments using remote servers (e.g. Vastai). note: ssh keys have to be setup already, along with torch/dependencies on a remote.
1. List the used server's username, ip, port, and name in a file with a format similar to `exmp_remotes.txt`.
2. Send the code to remotes: `./send_to_remotes.sh exmp_remotes.txt`
3. On the remote: 
   1. setup the environment: `./setup.sh`
   2. run the experiment for a specific shaping value and environment: `./start_runs.sh Pendulum-v1 td3 0.5`
6. when done training, copy the results back to the local machine: `./get_data_from_remotes.sh exmp_remotes.txt`