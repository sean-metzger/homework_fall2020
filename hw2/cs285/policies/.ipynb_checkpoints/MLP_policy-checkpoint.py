import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure import utils


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
#         with torch.no_grad():
            obs = np.asarray(obs)
            if len(obs.shape) > 1:
                observation = obs
            else:
                observation = obs[None]

    #         # Done todo return the action that the policy prescribes
    # #             print('selfdiscrete', self.discrete)
            if self.discrete: 
                observation = ptu.from_numpy(obs)
                possible_actions = self.logits_na(observation)
                probs = F.softmax(possible_actions)
                m = torch.distributions.categorical.Categorical(probs)
                action_to_take = m.sample()
                return ptu.to_numpy(action_to_take)
            else: 
                obs = ptu.from_numpy(obs)
                pred_mu = self.mean_net(obs)
                std = torch.exp(self.logstd)
                eps = torch.randn_like(pred_mu)
                pred = pred_mu + eps*std
                return ptu.to_numpy(pred)



    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # this was a todo. 
        if self.discrete: 
        # update that network .
            self.optimizer.zero_grad()

            pred = self.logits_na(observations)
            loss = F.cross_entropy(pred, actions)
            loss.backward()
            self.optimizer.step()

        else: 
            self.optimizer.zero_grad()
            pred_mu = self.mean_net(observations)
            # pred_logstd = self.logstd(observations)
            std = torch.exp(self.logstd)
            eps = torch.randn_like(pred_mu)
            pred = pred_mu + eps*std
            loss = F.MSELoss(pred, actions)
            loss.backward()
            self.optimizer.step()
            # update mean and std. 

        return loss.detach().cpu()

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
      # observation = torch.from_numpy(observation).float().to(ptu.device)
        if self.discrete: 
            logits = self.logits_na(observation)
            action_distribution = torch.distributions.categorical.Categorical(F.softmax(logits))
            return action_distribution
        else: 
            pred_mu = self.mean_net(observation)
            std = torch.exp(self.logstd)
            mvn = distributions.MultivariateNormal(pred_mu, scale_tril=torch.diag(std))
            return mvn

#     def forward(self, observation: torch.FloatTensor):
#         if self.discrete:
#             logits = self.logits_na(observation)
#             action_distribution = distributions.Categorical(logits=logits)
#             return action_distribution
#         else:
#             batch_mean = self.mean_net(observation)
#             scale_tril = torch.diag(torch.exp(self.logstd))
# #             print(batch_mean.shape, scale_tril.shape)
#             batch_dim = batch_mean.shape[0]
#             batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
#             action_distribution = distributions.MultivariateNormal(
#                 batch_mean,
#                 scale_tril=scale_tril,
#             )
#             return distributions.Normal(batch_mean, scale_tril)
#         return action_distribution

#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss
        
        
        
        self.optimizer.zero_grad()
        distribution = self.forward(observations)
        loss = -1*(distribution.log_prob(actions)*advantages).mean()
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            ## TODO: normalize the q_values to have a mean of zero and a standard deviation of one
            ## HINT: there is a `normalize` function in `infrastructure.utils`
            targets = utils.normalize(q_values, np.mean(q_values), np.std(q_values))
            targets = ptu.from_numpy(targets)

            ## TODO: use the `forward` method of `self.baseline` to get baseline predictions
            baseline_predictions = self.baseline(observations)
            
            ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            ## [ N ] versus shape [ N x 1 ]
            ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
            targets = torch.squeeze(targets)
            baseline_predictions = torch.squeeze(baseline_predictions)
            assert baseline_predictions.shape == targets.shape
            
            # TODO: compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
            # HINT: use `F.mse_loss`
            baseline_loss = F.mse_loss(baseline_predictions, targets)

            # TODO: optimize `baseline_loss` using `self.baseline_optimizer`
            # HINT: remember to `zero_grad` first
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }
        return train_log

    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]

