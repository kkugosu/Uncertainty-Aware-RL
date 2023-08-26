



[![Torch Version](https://img.shields.io/badge/torch>=1.10.0-61DAFB.svg?style=flat-square)](#torch)
[![Torchvision Version](https://img.shields.io/badge/torchvision>=0.2.2-yellow.svg?style=flat-square)](#torchvision)
[![Python Version](https://img.shields.io/badge/python->=3.6-blue.svg?style=flat-square)](#python)


# Uncertainty Aware RL


## 🎓 Guided Policy Search 

as the title, we learn policy guided from ilqr optimization.

we use bnn as dynamic model

overall process of algorithm is like this

$$ 1. \ randomly \ choose \ \pi_{ilqr} \ or \ \pi_\theta \ and \ implement. $$

$$ 2. \ learn \ dynamic \ by \ bnn $$

$$ 3. \ learn \ \pi_{ilqr} \ and \ \pi_\theta \ by \ using \ bnn $$

detail of process 3 is like below, dual gradient descent

first we set cost = f + $\lambda (constraint)$ which is lagrangian form

we name this cost as L($x^{*}(\lambda), \lambda$)

$x^{*}(\lambda)$ means trajectory $\tau $ and network parameter $\theta $

update rule is like this

$$1. \ \tau \leftarrow argmin_\tau L(\tau, \theta, \lambda) $$

$$2. \ \theta \leftarrow argmin_\theta L(\tau, \theta, \lambda) $$

$$3. \ \lambda \leftarrow \lambda  + \alpha * {dg \over d\lambda } $$

## 🌍 Experiment Environments

- **Cartpole**
- **Hopper**

## 📦 Requirements

- Gym
- Mujoco
- Python >= 3.8 
- Pytorch >= 1.12.0
- Numpy

## 📚 Papers & References

- **iLQR**: [TassaIROS12](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf)

- **MDGPS**: Reset-Free Guided Policy Search: Efficient Deep Reinforcement Learning with Stochastic Initial States

- **GPS**: [Guided Policy Search](https://jonathan-hui.medium.com/rl-guided-policy-search-gps-d1fae1084c24)

- **CS285**: Learning Neural Network Policies with Guided Policy Search under Unknown Dynamics

