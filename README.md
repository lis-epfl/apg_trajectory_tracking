# Drone Control with Analytic Policy Gradient

This repository contains the code accompanying the paper [*Learning to fly - Training Efficient Controllers via Analytic Policy Gradient*](https://arxiv.org/abs/2209.13052). We propose to combine the accuracy of Model Predictive Control with the efficiency (runtime) of learning-based approaches by training a controller with APG, i.e. by differentiating through the dynamics model:

![Learning paradigm](assets/paradigm.png)

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

Similarly, the cartpole or fixed wing drnoe can be trained (without generating any trajectories) with:
``` bash
python scripts/train_fixed_wing.py
python scripts/train_cartpole.py
```

### Evaluation

The trained models can be evaluated in a similar fashion, by running either of these commands:
``` bash
python scripts/evaluate_drone.py
python scripts/evaluate_fixed_wing.py
python scripts/evaluate_cartpole.py
```
