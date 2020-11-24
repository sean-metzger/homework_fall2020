from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import ipdb
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
        self.ws = torch.ones(n_dist, requires_grad=True)
        self.means = torch.randn((n_dist, self.ob_dim), requires_grad=True)
        self.stds = torch.zeros((n_dist, self.ob_dim), requires_grad=True)
            
        self.optimizer = torch.optim.SGD([self.means, self.ws, self.stds], lr=1e-3)
        self.ws.to(ptu.device)
        self.means.to(ptu.device)
        self.stds.to(ptu.device)
        
        self.c = torch.zeros(1, requires_grad=False)
        self.c.to(ptu.device)
        
    def get_distributions(self): 
        mix = dist.Categorical(self.ws.to(ptu.device))
        comp = dist.Independent(dist.Normal(self.means.to(ptu.device), torch.exp(self.stds.to(ptu.device) + 1e-12)), 1)
        gmm = dist.MixtureSameFamily(mix, comp)
        return gmm
        
    def forward(self, ob_no):
        
        ob_no = ob_no.to(ptu.device)
        self.c += ob_no.shape[0]
        ob_no = ob_no.to(ptu.device)
        gmm = self.get_distributions()
        log_probs = gmm.log_prob(ob_no)
   
        
        loss = -1*torch.mean(log_probs)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        gmm = self.get_distributions()
        log_probs_2 = gmm.log_prob(ob_no)
        
        if torch.sum(torch.isnan(self.means)) + torch.sum(torch.isnan(self.stds)) + torch.sum(torch.isnan(self.ws)).item() > 0:
            ipdb.set_trace()
        
        with torch.no_grad():
            pg = log_probs_2-log_probs
            bonus = 1/(torch.exp(.1*self.c.to(ptu.device).pow(-.5)*nn.functional.relu(pg)) -1)
            bonus = nn.functional.relu(bonus)
            bonus = 1/(torch.sqrt(bonus) + 1e-12)
            
        if torch.sum(torch.isnan(bonus)) + torch.sum(torch.isnan(pg)) > 0: 
            ipdb.set_trace()
              
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
        if torch.isnan(loss).item(): 
            ipdb.set_trace()
        return loss.item()
