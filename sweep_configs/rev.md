
Thank you for the accurate summary of our work and your helpful comments. 

In the "strengths" section, you mention "deriving clipping (albeit in specific circumstances when you already have pre-trained policies on related tasks)". We would like to emphasize that our results also hold when optimal (or even sub-optimal) policies are known in very different tasks. In section 5.1, the referenced $Q^{\pi_k}$ can be the value function corresponding to *any* policy, e.g. a suboptimal policy's value in an arbitrary task (with only the same state and action space). We therefore believe the results shown will be of a broader interest to the RL community, beyond the subset interested in compositionality.

We will now address your stated weaknesses:

1. We believe this work is of broader interest to the RL community. This is because we are able to extend our statements on compositional RL far beyond what was shown in previous work (we supply results in both standard and entropy-regularized RL for arbitrary function composition). In addition, our results hold for general (non-composition) value-based learning algorithms as shown in Theorem 5.6. We also provide exact results (Theorem 5.3) which were not previously available. The prior SOTA for a comparable result was a one-sided bound for a particular composition function [1].


2. In response to your comments and those of the other reviewers, we have begun such experiments on larger-scale tasks (MuJoCo environments). If provided the opportunity, we would happy to include the results of such experiments in a revised version of the paper.


3. While we agree that the focus of the current work is on such tasks (which differ only on the reward function), there is a path for future work that enables the extension to tasks with varying dynamics as well, through Adamczyk et. al.'s Theorem 8 and 9. The basic idea would be to use a $K$ function which incorporates the change in dynamics as well as the change in rewards to derive similar bounds. We agree that it would be of interest to explore the extension to more varied tasks in future work *but we believe this is beyond the scope of the current work*.

To address your comments:

- Indeed we have observed this behavior and looked into its downstream effects. We have found that such policies, although they are suboptimal in terms of a globally accurate value functions, they still provide a (looser) bound on the optimal value function of interest. We would like to note that this does not interfere with the theory developed, such a scenario is still applicable, but may generally lead to looser bounds. 

- We believe that the clipping approach developed would work in larger tasks. Although we have one experiment (Fig. 4) with FAs, the main purpose of the paper was to provide a general theoretical framework for bounds in RL with and without entropy regularization or composition. Experiments in compositional MuJoCo environments are currently in progress.

[1]: Haarnoja, T. et. al. 2018
