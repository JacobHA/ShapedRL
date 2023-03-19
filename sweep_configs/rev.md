Thank you for the detailed summary of our work and the questions. We'd first like to address a few points before turning to your posed weaknesses and questions.

Within the "strengths" section you have mentioned: "the provided experiments support well known facts, such as large learning rates is bad". We would like to emphasize that the referenced experiment actually shows two distinct phenomena: (a) as mentioned, large learning rates are bad and (b) large learning rates lead to higher frequencies of bound violations during training. We have shown in the Appendix, Fig. 9, that these conclusions are in agreement  (i.e. these metrics yield similar notions of "optimal" learning rate). Importantly, the clipping frequency, which is much easier to measure than the final trained policy's performance, acts as a good metric for "optimality" of the learning rate. We also believe that this sets the stage for future work investigating an adaptive learning rate schedule aimed at reducing the clipping frequency.

Thank you for introducing us to the work of [1] which we include in the updated version. 
OR composition in entropy-regularized RL:


$$ \kappa(s,a) = f(\{r(s,a)\}) + \gamma V_f(s') - f(\{Q_k(s,a)\})$$

$$ \kappa(s,a) = \max_k {r_k(s,a)} + \max_{a'} \max_k { Q_k(s',a') } - \max_k {Q_k(s,a)} $$

$$ \kappa(s,a) = \max_k {r_k(s,a)} + \max_{k} \max_{a'} { Q_k(s',a') } - \max_k {Q_k(s,a)} $$

$$ \kappa(s,a) = \max_k {r_k(s,a)} + \max_{k} \max_{a'} { Q_k(s',a') } - \max_k {Q_k(s,a)} $$

$$ \kappa(s,a) \geq \max_k (r_k(s,a) + \max_{a'} { Q_k(s',a') ) - \max_k {Q_k(s,a)} $$

$$ \kappa(s,a) \geq \max_k  Q_k(s,a)  - \max_k {Q_k(s,a)} = 0 $$


To address your raised weaknesses:

- We would argue that the ability to write the correction term as an *optimal* value function is more insightful than some other bounding function. This is because (a) the latter is more difficult to conceive of/derive in the general setting and (b) once the correction term is known to be an *optimal value function* it becomes straightforward to prove bounds on it, which transfers to bounds on the value function of interest. Additionally, more sophisticated bounds can be developed by using the equalities (e.g. Thm 1, Thm 5.6) as a starting point, depending on knowledge of dynamics or reward structure, for example.

- \textbf{To rebut this point: we have included several references to the work of Haarnoja which we have extended in multiple directions, as noted by reviewer 1}. In any case.. thank you for the suggestion. You are correct that we can now provide further intuition for the previously derived results. For example we can consider the case of linear convex-weighted combinations and also the exact results in entropy-regularized RL (van Niekerk). We expand on this by providing proof sketches of their results using our results...

- Although the focus of the current work is mainly theoretical, we agree that further experiments would strengthen the paper. Therefore, we have begun experiments in more complex settings. We have included the results of clipping on a non-compositional but continuous domain - MountainCar, and further compositional experiments in MuJoCo environments are a work in progress. \textbf{anything about stochasticity?}


Now, to address your questions:

- As an example, linear combinations as we have previously shown, and also we can show easily the OR gate (max composition) introduced by Boolean...

- (Spend some ink here explaining LMDP OR result) As you have mentioned, our bounds apply generally to any function. By restricting to classes of functions (e.g. non-convex weighted linear combinations), we can immediately obtain new bounds:

- ...
