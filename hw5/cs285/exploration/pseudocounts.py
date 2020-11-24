from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

import torch.distributions as dist

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class PseudoCounts(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # TODO: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        # WARNING: Make sure you use different types of weight 
        #          initializations for these two functions

        # HINT 1) Check out the method ptu.build_mlp
        # HINT 2) There are two weight init methods defined above
        
        n_dist = 4
        self.n_dist = n_dist
        
    def get_distributions(self): 
        
        
    def forward(self, ob_no):
        
        
        self.c += ob_no.shape[0]
        ob_no = ob_no.to(ptu.device)
        dists, weights = self.get_distributions()
        log_probs = 0
            
        log_probs = torch.sum(torch.log(log_probs + 1e-15), dim=-1)
        
            
        loss = -1*torch.mean(log_probs)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        dists, weights = self.get_distributions()
        log_probs_2 = 0
        for k in range(self.n_dist):
            log_probs_2 += weights[k]*torch.exp(dists[k].log_prob(ob_no))
            
        log_probs_2 = torch.sum(torch.log(log_probs_2 + 1e-15), dim=-1)
        
        pg = log_probs_2 - log_probs
        bonus = 1/(torch.exp(.1*self.c.to(ptu.device).pow(-.5)* nn.functional.relu(pg)) -1)
        bonus = nn.functional.relu(bonus)
        bonus = torch.sqrt(bonus)
        
        return bonus

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # TODO: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        ob_no = ptu.from_numpy(ob_no)
        loss = self(ob_no).mean()
        return loss.item()
