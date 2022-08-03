import torch
import random
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Policy:
    def __init__(self, policy, model, converter):
        self.policy = policy
        self.model = model
        self.converter = converter
        self.softmax = nn.Softmax(dim=-1)

    def select_action(self, n_p_o):
        t_p_o = torch.tensor(n_p_o, device=device, dtype=torch.float32)

        if self.policy == "gps":
            #half ilqg
            #half global policy
            with torch.no_grad():
                probability = self.model(t_p_o)

            t_a_index = torch.multinomial(probability, 1)
            n_a = self.converter.index2act(t_a_index.squeeze(-1), 1)
            return n_a

        else:
            print("model name error")
            return None
