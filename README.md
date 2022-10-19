# Training Efficient Controllers via Analytic Policy Gradient

This repository contains the code accompanying the paper **Training Efficient Controllers via Analytic Policy Gradient** ([PDF](https://arxiv.org/abs/2209.13052)) by [Nina Wiedemann*](https://github.com/NinaWie/), [Valentin Wüest*](https://github.com/vwueest/), [Antonio Loquercio](https://antonilo.github.io/), [Matthias Müller](https://matthias.pw/), [Dario Floreano](https://people.epfl.ch/dario.floreano), and [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html). We propose to combine the accuracy of Model Predictive Control with the efficiency (runtime) of learning-based approaches by training a controller with APG, i.e. by differentiating through the dynamics model:

![Learning paradigm](assets/paradigm.png)

For an overview of our method and trajectory tracking examples, check out our [video](https://arxiv.org/src/2209.13052v1/anc/arxiv_video.mp4).

If you use any of this code, please cite the following publication:

```bibtex
@Article{wiedemannwueest2022training,
  title   = {Training Efficient Controllers via Analytic Policy Gradient},
  author  = {Wiedemann, Nina and W{\"u}est, Valentin and Loquercio, Antonio and M{\"u}ller, Matthias and Floreano, Dario and Scaramuzza, Davide},
  journal = {arXiv preprint arXiv:2209.13052},
  year    = {2022}
}
```

## Abstract

Control design for robotic systems is complex and often requires solving an optimization to follow a trajectory accurately. Online optimization approaches like Model Predictive Control (MPC) have been shown to achieve great tracking performance, but require high computing power. Conversely, learning-based offline optimization approaches, such as Reinforcement Learning (RL), allow fast and efficient execution on the robot but hardly match the accuracy of MPC in trajectory tracking tasks. In systems with limited compute, such as aerial vehicles, an accurate controller that is efficient at execution time is imperative. We propose an Analytic Policy Gradient (APG) method to tackle this problem. APG exploits the availability of differentiable simulators by training a controller offline with gradient descent on the tracking error. We address training instabilities that frequently occur with APG through curriculum learning and experiment on a widely used controls benchmark, the CartPole, and two common aerial robots, a quadrotor and a fixed-wing drone. Our proposed method outperforms both model-based and model-free RL methods in terms of tracking error. Concurrently, it achieves similar performance to MPC while requiring more than an order of magnitude less computation time. Our work provides insights into the potential of APG as a promising control method for robotics. To facilitate the exploration of APG, we open-source our code.

## Installation

Install all requirements in a virtual environment with:
``` bash
python -m venv env
source env/bin/activate
cd apg_drone_control
pip install -e .
```

### Training

To train a controller for the quadrotor, we first need to create random polynomial trajectories as train and test data. Run:
``` bash
python scripts/generate_trajectories.py
```

Then, you can start training:
``` bash
python scripts/train_drone.py
```

Similarly, the cartpole or fixed wing drone can be trained (without generating any trajectories) with:
``` bash
python scripts/train_fixed_wing.py
python scripts/train_cartpole.py
```

See our [training documentation](training_details.pdf) for further information.

### Evaluation

The trained models can be evaluated in a similar fashion, by running either of these commands:
``` bash
python scripts/evaluate_drone.py -a 50
python scripts/evaluate_fixed_wing.py -a 50
python scripts/evaluate_cartpole.py -a 50
```

Notes:
* The flag `-a` determines the number of iterations.
* With the animate-flag an animation is shown (`python scripts/evaluate_drone.py --animate` and analogous for fixed wing and cartpole).

**Baseline comparison:**

1) MPC: 
The MPC baseline is integrated in the evaluate-scripts. Simply specify mpc with the model flag, e.g. `python scripts/evaluate_drone.py -m mpc -a 50`.

2) PPO:
PPO models can be trained and evaluated with the script `baselines/ppo_baseline.py` (e.g. `python baselines/ppo_baseline.py -r fixed_wing -a 50`).

3) PETS:
PETS training is done with the [mbrl](https://github.com/facebookresearch/mbrl-lib) library provided by Facebook Research. Our code can be found in the script `baselines/pets_baseline.py`.
