# Drone Control with Analytic Policy Gradient

This repository contains the code accompanying the paper *Learning to fly - Training Efficient Controllers via Analytic Policy Gradient*. We propose to combine the accuracy of Model Predictive Control with the efficiency (runtime) of learning-based approaches by training a controller with APG, i.e. by differentiating through the dynamics model:

![Learning paradigm](assets/paradigm.png)

Install all requirements in a virtual environment with 

```
python -m venv env
source env/bin/activate
cd apg_drone_control
pip install -e .
```

### Training

To train a controller for the quadrotor, we first need to create random polynomial trajectories as train and test data. Run
```
python train_scripts/generate_trajectories.py
```

Then, you can start training:
```
python train_scripts/train_drone.py
```

Similarly, the cartpole or fixed wing drnoe can be trained (without generating any trajectories) with
```
python train_scripts/train_fixed_wing.py
python train_scripts/train_cartpole.py
```