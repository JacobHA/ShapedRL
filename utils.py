import random

import numpy as np


def sample_wandb_hyperparams(params, int_hparams=None):
    sampled = {}
    for k, v in params.items():
        if 'values' in v:
            sampled[k] = random.choice(v['values'])
        elif 'distribution' in v:
            if v['distribution'] == 'uniform' or v['distribution'] == 'uniform_values':
                sampled[k] = random.uniform(v['min'], v['max'])
            elif v['distribution'] == 'normal':
                sampled[k] = random.normalvariate(v['mean'], v['std'])
            elif v['distribution'] == 'log_uniform_values':
                emin, emax = np.log(v['max']), np.log(v['min'])
                sample = np.exp(random.uniform(emin, emax))
                sampled[k] = sample
            else:
                raise NotImplementedError(f"Distribution sampling not implemented for {v['distribution']}, required by {k}")
        else:
            raise NotImplementedError(f"Unsupported hparam sampling range format in {k}")
        if k in int_hparams:
            sampled[k] = int(sampled[k])
    return sampled


