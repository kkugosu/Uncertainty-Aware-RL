# MB_RL

guided policy search 

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

$$3. \ \lambda \leftarrow \lambda  + \alpha * (dg \over d\lambda ) $$

* * * 

repo

ilqr

https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf

mdgps

Reset-Free Guided Policy Search: Efficient Deep Reinforcement
Learning with Stochastic Initial States

gps

https://jonathan-hui.medium.com/rl-guided-policy-search-gps-d1fae1084c24

cs285

“Learning Neural Network Policies with Guided Policy
Search under Unknown Dynamics
