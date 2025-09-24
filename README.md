Thank you for visiting our repository for "BootStrapped Reward Shaping" (BSRS) 

### Run experiments based on globally set variables:
Specify the experiment parameters in the config.py file. (No arguments are passed to the experiment.py file)
The wandb config files for each algo are in the sweep_configs folder.

```
python experiment.py
```


If you would like to cite our work, please use the following bibtex based on the AAAI proceedings:
```
@inproceedings{adamczyk2025bootstrapped,
  title={Bootstrapped Reward Shaping},
  author={Adamczyk, Jacob and Makarenko, Volodymyr and Tiomkin, Stas and Kulkarni, Rahul V},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={15},
  pages={15302--15310},
  year={2025}
}
```
# Contributions
Contributions are always welcome! Feel free to add your own algorithm with bootstrapped reward shaping. The general idea applies to any RL algorithm that has access to `V(s)` the state-value function, for calculating the potential function.

## TODO:
- [ ] Add pip / poetry installable setup

# Future Work / Ideas:
- (Learned) schedule for eta as training progresses
- eta(s) - state-dependent theory holds
- Improve the bounds
