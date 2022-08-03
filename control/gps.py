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
import random
import torch.onnx as onnx
GAMMA = 0.98


class GPS(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.Dynamics = bayesian_nn.BayesianModel(self.o_s + self.a_s, self.h_s, self.o_s).to(self.device)
        self.Reward = NN.ValueNN(self.o_s, self.h_s, self.a_s**2 + self.a_s + 1).to(self.device)
        self.R_NAF = converter.NAF(self.o_s, self.a_s, self.Reward)
        self.Policy_net = NN.ValueNN(self.o_s, self.h_s, self.a_s**2 + self.a_s).to(self.device)
        self.P_NAF = converter.NAFGaussian(self.o_s, self.a_s, self.Policy_net)
        
        self.ilqg = ilqr.ilqr(ts, dyn, re, sl, al, b_s)
        self.policy = policy.Policy(self.cont, self.Dynamics, self.converter)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=self.e_trace, done_penalty=self.d_p)
        self.optimizer_D = torch.optim.SGD(self.Dynamics.parameters(), lr=self.lr)
        self.optimizer_R = torch.optim.SGD(self.Reward.parameters(), lr=self.lr)
        self.optimizer_P = torch.optim.SGD(self.Policy.parameters(), lr=self.lr)
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
            self.Policy.load_state_dict(torch.load(self.PARAM_PATH + "/p.pth"))

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
            torch.save(self.Policy.state_dict(), self.PARAM_PATH + "/p.pth")

        for param in self.Dynamics.parameters():
            print("----------dyn-------------")
            print(param)
        for param in self.Reward.parameters():
            print("----------rew--------------")
            print(param)
        for param in self.Policy.parameters():
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

            predict_r = self.R_NAF.get_batch_reward(t_o, t_a)
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
        queue_loss = None
        policy_loss = None
        while i < self.m_i:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            t_a_index = self.converter.act2index(n_a, self.b_s).unsqueeze(axis=-1)
            t_o = torch.tensor(n_o, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            t_p_weight = torch.gather(self.updatedPG(t_p_o), 1, t_a_index)
            t_p_qvalue = torch.gather(self.updatedDQN(t_p_o), 1, t_a_index)
            weight = torch.transpose(torch.log(t_p_weight), 0, 1)
            policy_loss = -torch.matmul(weight, t_p_qvalue)/self.b_s
            t_trace = torch.tensor(n_d, dtype=torch.float32).to(self.device).unsqueeze(-1)

            with torch.no_grad():
                n_a_expect = self.policy.select_action(n_o)
                t_a_index = self.converter.act2index(n_a_expect, self.b_s).unsqueeze(-1)
                t_qvalue = torch.gather(self.baseDQN(t_o), 1, t_a_index)
                t_qvalue = t_qvalue*(GAMMA**t_trace) + t_r.unsqueeze(-1)

            queue_loss = self.criterion(t_p_qvalue, t_qvalue)

            self.optimizer_p.zero_grad()
            policy_loss.backward(retain_graph=True)
            for param in self.updatedPG.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_p.step()

            self.optimizer_q.zero_grad()
            queue_loss.backward()
            for param in self.updatedDQN.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            i = i + 1
        print("loss1 = ", policy_loss)
        print("loss2 = ", queue_loss)

        return policy_loss, queue_loss