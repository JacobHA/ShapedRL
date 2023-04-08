### Run experiments based on globally set variables

Specify the experiment parameters in the config.py file. (No arguments are passed to the experiment.py file)
The wandb config files for each algo are in the sweep_configs folder.

```
python experiment.py
```

# Requirements:
- pip install opencv-python
- pip install gym[atari]

# TODOs:
- [x] Add a ShapedTD3 class
- [x] Automatically choose the sweep config.yml file based on model used
- [x] Clean up sweep id configuration
- [ ] Clean up eval callback 
- [ ] Make tests for missing args in hparam (e.g. DQN() when "policy" is missing from hparams)
- [ ] SpaceInvaders needs frameskip=3 (see Sec. 5 of https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

# Pitfalls:
- [ ] I had to go to env/lib/python3.8/site-packages/stable_baselines3/common/atari_wrappers.py L36 and change it to noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)


# Related Work:
- https://arxiv.org/pdf/2011.02669.pdf
- Munchausen: https://arxiv.org/abs/2007.14430
- (related to above: https://arxiv.org/abs/2205.07467)

# Future Work / Ideas:
- For automatically adjusting the learning rate (to min. clipping) or adjust the soft-clip weight parameter:
- Sec 5 in https://arxiv.org/pdf/1812.05905.pdf

# Hyperparameter Sources
- Acrobot: https://huggingface.co/sgoodfriend/dqn-Acrobot-v1
