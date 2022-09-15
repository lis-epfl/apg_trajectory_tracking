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