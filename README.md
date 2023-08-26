



[![Torch Version](https://img.shields.io/badge/torch>=1.10.0-61DAFB.svg?style=flat-square)](#torch)
[![Torchvision Version](https://img.shields.io/badge/torchvision>=0.2.2-yellow.svg?style=flat-square)](#torchvision)
[![Python Version](https://img.shields.io/badge/python->=3.6-blue.svg?style=flat-square)](#python)


# Uncertainty Aware RL


## 🎓 Guided Policy Search 

As suggested by the title, we guide the policy search based on iLQR optimization. We use a Bayesian Neural Network (BNN) as the dynamic model.

**Overall Process of Algorithm:**

1. Randomly choose either \(\pi_{ilqr}\) or \(\pi_\theta\) and implement.
2. Learn dynamics using the BNN.
3. Learn \(\pi_{ilqr}\) and \(\pi_\theta\) using the BNN.

**Details of Process 3:** This follows the Dual Gradient Descent method. First, we set the cost as:

\[ \text{cost} = f + \lambda \times \text{(constraint)} \]

This is in the Lagrangian form and we denote this cost as \( L(x^{*}(\lambda), \lambda) \). Here, \( x^{*}(\lambda) \) represents both the trajectory \( \tau \) and the network parameter \( \theta \). The update rule is:

1. \( \tau \leftarrow \text{argmin}_\tau L(\tau, \theta, \lambda) \)
2. \( \theta \leftarrow \text{argmin}_\theta L(\tau, \theta, \lambda) \)
3. \( \lambda \leftarrow \lambda + \alpha \times \frac{dg}{d\lambda} \)

---

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

