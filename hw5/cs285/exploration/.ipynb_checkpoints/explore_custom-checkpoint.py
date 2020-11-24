from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class ContrastiveQueue(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec
        self.prev_obno = None

        # TODO: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        # WARNING: Make sure you use different types of weight 
        #          initializations for these two functions

        # HINT 1) Check out the method ptu.build_mlp
        # HINT 2) There are two weight init methods defined above
        

        self.encoder_q = ptu.build_mlp(input_size=self.ob_dim, 
                               output_size=self.output_size, 
                               n_layers= self.n_layers,
                               size = self.size, 
                               init_method = init_method_1)
        
        self.encoder_k = ptu.build_mlp(input_size=self.ob_dim, 
                               output_size=self.output_size, 
                               n_layers= self.n_layers,
                               size = self.size, 
                               init_method = init_method_2)
        
        self.optimizer = self.optimizer_spec.constructor(
            self.encoder_k.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.encoder_q.to(ptu.device)
        self.encoder_k.to(ptu.device)
        
        
        print('out', self.output_size)
        self.K = 65536//2
        dim = self.output_size
        
        self.temp = .2
        
        self.queue =  torch.randn(dim, self.K)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue.to(ptu.device)
        print('ptu device', ptu.device)
        
        self.ptr = torch.zeros(1, dtype=torch.long)
        
        
    def forward(self, ob_no):
        
        q = self.encoder_q(ob_no)
        q = nn.functional.normalize(q, dim=1)
        
        import time
        s = time.time()
        logits = torch.einsum('nc, ck->nk',[q.detach().cpu(), self.queue.clone().detach()]) # from moco source. 
        
#         print('einsum time', time.time()-s)
        logits /= self.temp
        error = torch.logsumexp(logits, dim=1)
        return error
    

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    
    @torch.no_grad()
    def update(self, ob_no):
        # Just add a sample to the queue. 
        ob_no = ptu.from_numpy(ob_no)
        q = self.encoder_q(ob_no)
        q = nn.functional.normalize(q, dim=1)
        self.queue[:, self.ptr:self.ptr+ob_no.shape[0]] = q.T
        self.ptr = (self.ptr+ob_no.shape[0]) % self.K
        
        
        loss = torch.zeros(1).to(ptu.device)
        return loss.item()
