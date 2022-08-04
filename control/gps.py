from control import BASE, policy
import gym
import torch
import numpy as np
import sys
from torch import nn
from NeuralNetwork import NN, bayesian_nn
from utils import buffer
import ilqr
from utils import converter
GAMMA = 0.98


class GPS(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.Dynamics = bayesian_nn.BayesianModel(self.o_s + self.a_s, self.h_s, self.o_s).to(self.device)
        self.Reward = NN.ValueNN(self.o_s, self.h_s, self.a_s**2 + self.a_s + 1).to(self.device)
        self.R_NAF = converter.NAFReward(self.o_s, self.a_s, self.Reward)
        self.Policy_net = NN.ValueNN(self.o_s, self.h_s, self.a_s**2 + self.a_s).to(self.device)
        self.P_NAF = converter.NAFPolicy(self.o_s, self.a_s, self.Policy_net)

        self.iLQG = ilqr.IterativeLQG(self.Dynamics, self.R_NAF, self.P_NAF, self.o_s, self.a_s)
        self.policy = policy.Policy(self.cont, self.P_NAF, self.converter)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=self.e_trace, done_penalty=self.d_p)
        self.optimizer_D = torch.optim.SGD(self.Dynamics.parameters(), lr=self.lr)
        self.optimizer_R = torch.optim.SGD(self.Reward.parameters(), lr=self.lr)
        self.optimizer_P = torch.optim.SGD(self.Policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='mean')

    def get_policy(self):
        return self.policy

    def reward_naf(self):
        self.Reward = NN.ValueNN(self.o_s, self.h_s, self.a_index_s).to(self.device)

    def training(self, load=int(0)):

        if int(load) == 1:
            print("loading")
            self.Dynamics.load_state_dict(torch.load(self.PARAM_PATH + "/d.pth"))
            self.Reward.load_state_dict(torch.load(self.PARAM_PATH + "/r.pth"))
            self.Policy_net.load_state_dict(torch.load(self.PARAM_PATH + "/p.pth"))

            print("loading complete")
        else:
            pass
        i = 0
        while i < self.t_i:
            print(i)

            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            dyn_loss, rew_loss = self.train_dynamic_per_buff()
            policy_loss = self.train_policy_per_buff()
            self.writer.add_scalar("dyn/loss", dyn_loss, i)
            self.writer.add_scalar("rew/loss", rew_loss, i)
            self.writer.add_scalar("rew/loss", policy_loss, i)
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            torch.save(self.Dynamics.state_dict(), self.PARAM_PATH + "/d.pth")
            torch.save(self.Reward.state_dict(), self.PARAM_PATH + "/r.pth")
            torch.save(self.Policy_net.state_dict(), self.PARAM_PATH + "/p.pth")

        for param in self.Dynamics.parameters():
            print("----------dyn-------------")
            print(param)
        for param in self.Reward.parameters():
            print("----------rew--------------")
            print(param)
        for param in self.Policy_net.parameters():
            print("----------pol--------------")
            print(param)
        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_dynamic_per_buff(self):
        i = 0
        dyn_loss = None
        rew_loss = None
        while i < self.m_i:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            t_a = torch.tensor(n_a, dtype=torch.float32).to(self.device)
            t_o = torch.tensor(n_o, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            sa_in = torch.cat((t_p_o, t_a), dim=-1)
            predict_o = self.Dynamics(sa_in)
            dyn_loss = self.criterion(t_o, predict_o) + self.Dynamics.kld_loss()

            predict_r = self.R_NAF.sa_reward(sa_in)
            rew_loss = self.criterion(t_r, predict_r)

            self.optimizer_D.zero_grad()
            dyn_loss.backward(retain_graph=True)
            for param in self.Dynamics.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_D.step()

            self.optimizer_R.zero_grad()
            rew_loss.backward()
            for param in self.Reward.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_R.step()

            i = i + 1
        print("loss1 = ", dyn_loss)
        print("loss2 = ", rew_loss)

        return dyn_loss, rew_loss

    def train_policy_per_buff(self):
        i = 0
        timestep = 5
        lamb = 1
        kld = 0
        self.iLQG.set_initialize(self.b_s, timestep)
        while i < self.m_i:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                mean, var = self.iLQG.fit(self.R_NAF, self.P_NAF)
            # update local policy all we need is action(mean) and var
            t_mean, t_var = self.P_NAF.prob(t_p_o)
            mean_d = mean - t_mean
            mean_d_t = torch.transpose(mean_d, -2, -1)
            kld = torch.log(torch.linalg.det(t_var) - torch.linalg.det(var)) +\
                  torch.trace(torch.matmul(torch.linalg.inv(t_var), var)) + \
                  torch.matmul(torch.matmul(mean_d, torch.linalg.inv(t_var)), mean_d_t)
            # kl - divergence - between - two - multivariate - gaussians
            self.optimizer_P.zero_grad()
            kld.backward(retain_graph=True)
            for param in self.Policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_P.step()
            # update global policy

            lamb = lamb + kld
            # update lambda

            i = i + 1
        print("policy loss = ", kld)

        return kld
