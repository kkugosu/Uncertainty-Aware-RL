import numpy as np
import torch
from torch import nn
import torch.autograd.functional as Fu
from functorch import vmap, hessian, jacfwd
STATELEN = 5
ACTLEN = 3
STEP_SIZE = 4

# based on https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
# torch.distributions.multivariate_normal.MultivariateNormal
# MultivariateNormal(torch.zeros(2), torch.eye(2)) #psd

class ilqr:

    def __init__(self, ts, dyn, re, sl, al, b_s):
        """
        Args:
            ts: time step
            dyn: dynamic
            re: reward
            sl: state length
            al: action length
        """
        self.ts = ts
        self.dyn = dyn
        self.re = re
        self.sl = sl
        self.al = al
        self.b_s = b_s

        self.S = torch.zeros((self.ts, self.b_s, self.sl))
        self.A = torch.zeros((self.ts, self.b_s, self.al))
        self.R = torch.empty((self.ts, self.b_s, 1))
        self.K_arr = torch.zeros(self.ts, self.b_s, self.al, self.sl)
        self.k_arr = torch.zeros(self.ts, self.b_s, 1, self.al)
        self.ifconv = 0



    def _get_act_prob(self):

    def _forward(self):

        new_S = torch.zeros((self.ts, self.b_s, self.sl))
        new_A = torch.zeros((self.ts, self.b_s, self.al))
        s = self.S[0].clone().detach()
        # for p in self.dyn.parameters():
        #    print(p)
        i = 0
        while i < self.ts:
            new_S[i] = s
            state_difference = (new_S[i] - self.S[i]).unsqueeze(1)

            state_action_trans = torch.matmul(state_difference, torch.transpose((self.K_arr[i]), 1, 2))

            new_A[i] = (state_action_trans + self.k_arr[i]).squeeze(1) + self.A[i]
            sa_in = torch.cat((new_S[i], new_A[i]), dim=1)

            s = self.dyn(sa_in)

            # state shape = [1,state_size]

            self.R[i] = self._get_batch_reward(sa_in)

            i = i + 1
        self.S = new_S
        self.A = new_A

    def _backward(self):

        C = torch.zeros(self.b_s, self.al + self.sl, self.al + self.sl)
        F = torch.zeros(self.b_s, self.sl, self.al + self.sl)
        c = torch.zeros(self.b_s, 1, self.al + self.sl)
        V = torch.zeros(self.b_s, self.sl, self.sl)
        v = torch.zeros(self.b_s, 1, self.sl)
        sa_in = torch.cat((self.S, self.A), dim=-1)

        i = self.ts - 1
        while i > -1:

            C = vmap(hessian(self._get_reward))(sa_in[i])
            # shape = [state+action, state+action]
            # print(torch.sum(C[j]))
            c = vmap(jacfwd(self._get_reward))(sa_in[i])

            # shape = [1, state+action]
            # print(torch.sum(c[j]))
            F = vmap(jacfwd(self.dyn))(sa_in[i])
            # shape = [state, state+action]
            # print(torch.sum(F[j]))
            # use jacfwd because input is large than output
            transF = torch.transpose(F, 1, 2)
            Q = C + torch.matmul(torch.matmul(transF, V), F)

            # eq 5[c~e]
            q = c.unsqueeze(1) + torch.matmul(v, F)

            # eq 5[a~b]

            Q_pre1, Q_pre2 = torch.split(Q, [self.sl, self.al], dim=1)
            Q_xx, Q_xu = torch.split(Q_pre1, [self.sl, self.al], dim=2)
            Q_ux, Q_uu = torch.split(Q_pre2, [self.sl, self.al], dim=2)

            Q_x, Q_u = torch.split(q, [self.sl, self.al], dim=-1)
            ## how to batched eye?
            # print(Q_uu)
            try:
                invQuu = torch.linalg.inv(Q_uu - torch.eye(self.al) * 0.01)  # - torch.eye(self.al)) #regularize term
                # eq [9]
            except:
                invQuu = torch.linalg.inv(Q_uu + torch.eye(self.al) * 0.01)
                self.ifconv = 1

            K = -torch.matmul(invQuu, Q_ux)
            transK = torch.transpose(K, 1, 2)
            # K_t shape = [actlen, statelen]

            k = -torch.matmul(Q_u, invQuu)
            # k_t shape = [1,actlen]

            V = (Q_xx + torch.matmul(Q_xu, K) +
                 torch.matmul(transK, Q_ux) +
                 torch.matmul(torch.matmul(transK, Q_uu), K)
                 )
            # eq 11c
            # V_t shape = [statelen, statelen]

            v = (Q_x + torch.matmul(k, Q_ux) +
                 torch.matmul(Q_u, K) +
                 torch.matmul(k, torch.matmul(Q_uu, K))
                 )
            # eq 11b
            # v_t shape = [1, statelen]

            self.K_arr[i] = K
            self.k_arr[i] = k
            i = i - 1

    def fit(self, action, state):
        self.A = action
        self.S[0] = state[0]
        setattr(self.dyn, 'freeze', 1)
        i = 0
        while (self.ifconv != 1) and i < 100:
            i = i + 1
            self._forward()
            self._backward()

            print("act", self.A)
        return self.A

# for param in rew.parameters():
#    print(param)
my_Dyna = Dynamics(STATELEN + ACTLEN, STATELEN, STATELEN)
my_reward = reward(STATELEN, STATELEN , ACTLEN**2 + ACTLEN + 1)
TIME_STEP = 6
BATCH_SIZE = 5
myilqr = ilqr(TIME_STEP, my_Dyna, my_reward, STATELEN, ACTLEN, BATCH_SIZE)
action = torch.rand((TIME_STEP, BATCH_SIZE, ACTLEN))
state = torch.zeros((TIME_STEP, BATCH_SIZE, STATELEN))
#simulate
#fit reward, dynamic
myilqr.fit(action, state)