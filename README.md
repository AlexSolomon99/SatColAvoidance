# SatColAvoidance

This repo contains the modules required to train an agent to manoeuvre a satellite in case of a collision event. 
This repo only contains training and evaluation pipelines for the agent, with the policies being trained according to 3 
different algorithms:
#### 1. PPO
#### 2. DQN
#### 3. REINFORCE

For a more detailed description of the analysis done in this repo, please check out our paper here:
[Collision Avoidance and Return Manoeuvre Optimisation for
Low-Thrust Satellites Using Reinforcement Learning](https://www.scitepress.org/Papers/2025/132490/132490.pdf) 

Corresponding Environment
------------------
The environment which the current repo is referencing is https://github.com/AlexSolomon99/SatColAvoidEnv.
To install it using pip, use the following command:

**Environment installation**: pip install -i https://test.pypi.org/simple/ gym-satellite-ca==0.2.0


Citing Us
------------------

If you reference or use the Collision Avoidance Environment in your research, please cite us:

```
@conference{icaart25,
author={Alexandru Solomon and Ciprian Paduraru},
title={Collision Avoidance and Return Manoeuvre Optimisation for Low-Thrust Satellites Using Reinforcement Learning},
booktitle={Proceedings of the 17th International Conference on Agents and Artificial Intelligence - Volume 3: ICAART},
year={2025},
pages={1009-1016},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0013249000003890},
isbn={978-989-758-737-5},
issn={2184-433X},
}
```