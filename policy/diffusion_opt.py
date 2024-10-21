import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR
from .helpers import (
    Losses
)
class DiffusionOPT(BasePolicy):

    def __init__(
            self,
            state_dim: int,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            # dist_fn: Type[torch.distributions.Distribution],
            device: torch.device,
            tau: float = 0.005,
            gamma: float = 1,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            bc_coef: bool = False,
            exploration_noise: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        # Initialize actor network and optimizer if provided
        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor  # Actor network
            self._target_actor = deepcopy(actor)  # Target actor network for stable learning
            self._target_actor.eval()  # Set target actor to evaluation mode
            self._actor_optim: torch.optim.Optimizer = actor_optim  # Optimizer for the actor network
            self._action_dim = action_dim  # Dimensionality of the action space

        # Initialize critic network and optimizer if provided
        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic  # Critic network
            self._target_critic = deepcopy(critic)  # Target critic network for stable learning
            self._critic_optim: torch.optim.Optimizer = critic_optim  # Optimizer for the critic network
            self._target_critic.eval()  # Set target critic to evaluation mode

        # If learning rate decay is applied, initialize learning rate schedulers for both actor and critic
        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt, eta_min=0.)

        # Initialize other parameters and configurations
        self._tau = tau  # Soft update coefficient for target networks
        self._gamma = gamma  # Discount factor for future rewards
        self._rew_norm = reward_normalization  # If true, normalize rewards
        self._n_step = estimation_step  # Steps for n-step return estimation
        self._lr_decay = lr_decay  # If true, apply learning rate decay
        self._bc_coef = bc_coef  # Coefficient for policy gradient loss
        self._device = device  # Device to run computations on
        self.noise_generator = GaussianNoise(sigma=exploration_noise)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        # Compute the actions for next states with target actor network
        ttt = self(batch, model='_target_actor', input='obs_next').act
        # Evaluate these actions with target critic network
        batch.obs_next = to_torch(batch.obs_next, device=self._device, dtype=torch.float32)
        target_q = self._target_critic.q_min(batch.obs_next, ttt)
        return target_q  # return the minimum of the dual Q values

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        # Compute n-step return for transitions in the batch
        return self.compute_nstep_return(
            batch,
            buffer,
            indices,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm
        )

    def update(
            self,
            sample_size: int,
            buffer: Optional[ReplayBuffer],
            **kwargs: Any
    ) -> Dict[str, Any]:
        # If no replay buffer is provided, return an empty dictionary
        if buffer is None: return {}
        self.updating = True # Indicate that the policy is being updated

        # Sample a batch of transitions from replay buffer
        batch, indices = buffer.sample(sample_size)

        # Compute n-step return for the sampled transitions
        batch = self.process_fn(batch, buffer, indices)
        # Update network parameters
        result = self.learn(batch, **kwargs)
        if self._lr_decay: # If learning rate decay is enabled, step the learning rate schedulers
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        self.updating = False # Indicate that the policy update has finished
        return result

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            model: str = "actor"
    ) -> Batch:
        # Convert batch observations to PyTorch tensors
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        # Use actor or target actor based on provided model argument
        model_ = self._actor if model == "actor" else self._target_actor
        # Feed observations through the selected model to get action logits
        logits, hidden = model_(obs_), None

        if self._bc_coef:
            acts = logits
        else:
            if np.random.rand() < 0.1:
                # Add exploration noise to the actions
                noise = to_torch(self.noise_generator.generate(logits.shape),
                                 dtype=torch.float32, device=self._device)
                # Add the noise to the action
                acts = logits + noise
                acts = torch.clamp(acts, -1, 1)
            else:
                acts = logits

        dist = None  # does not use a probability distribution for actions

        return Batch(logits=logits, act=acts, state=obs_, dist=dist)

    def _to_one_hot(
            self,
            data: np.ndarray,
            one_hot_dim: int
    ) -> np.ndarray:
        # Convert the provided data to one-hot representation
        batch_size = data.shape[0]
        one_hot_codes = np.eye(one_hot_dim)
        # print(data[1])
        one_hot_res = [one_hot_codes[data[i]].reshape((1, one_hot_dim))
                       for i in range(batch_size)]
        return np.concatenate(one_hot_res, axis=0)

    def _update_critic(self, batch: Batch) -> torch.Tensor:
        # Compute the critic's loss and update its parameters
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(batch.act, device=self._device, dtype=torch.float32)
        target_q = batch.returns # Target Q values are the n-step returns
        # print('target_q',target_q)
        # td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        current_q1, current_q2 = self._critic(obs_,acts_) # Current Q values are the critic's output
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q) # Compute the MSE loss
        # critic_loss = F.mse_loss(current_q1, target_q)

        self._critic_optim.zero_grad() # Zero the critic optimizer's gradients
        critic_loss.backward() # Backpropagate the loss
        self._critic_optim.step() # Perform a step of optimization
        return critic_loss


    def _update_bc(self, batch: Batch, update: bool = False) -> torch.Tensor:
        # Compute the behavior cloning loss
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        # expert_actions = torch.Tensor([info["sub_expert_action"] for info in batch.info]).to(self._device)
        expert_actions = torch.Tensor([info["expert_action"] for info in batch.info]).to(self._device)

        bc_loss = self._actor.loss(expert_actions, obs_).mean()

        if update:  # Update actor parameters if update flag is True
            self._actor_optim.zero_grad()  # Zero the actor optimizer's gradients
            bc_loss.backward()  # Backpropagate the loss
            self._actor_optim.step()  # Perform a step of optimization
        return bc_loss

    def _update_policy(self, batch: Batch, update: bool = False) -> torch.Tensor:
        # Compute the policy gradient loss
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(self(batch).act, device=self._device, dtype=torch.float32)
        pg_loss = - self._critic.q_min(obs_, acts_).mean()
        if update:
            self._actor_optim.zero_grad()
            pg_loss.backward()
            self._actor_optim.step()
        return pg_loss

    def _update_targets(self):
        # Perform soft update on target actor and target critic. Soft update is a method of slowly blending
        # the regular and target network to provide more stable learning updates.
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    def learn(
            self,
            batch: Batch,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        # Update critic network. The critic network is updated to minimize the mean square error loss
        # between the Q-value prediction (current_q1) and the target Q-value (target_q).
        critic_loss = self._update_critic(batch)
        # Update actor network. Here, we first calculate the policy gradient (pg_loss) and
        # behavior cloning loss (bc_loss) but we do not update the actor network yet.
        # The overall loss is a weighted combination of policy gradient loss and behavior cloning loss.
        if self._bc_coef:
            bc_loss = self._update_bc(batch, update=False)
            overall_loss = bc_loss
        else:
            pg_loss = self._update_policy(batch, update=False)
            overall_loss = pg_loss

        self._actor_optim.zero_grad()
        overall_loss.backward()
        self._actor_optim.step()

        # Update the target networks
        self._update_targets()
        return {
            'loss/critic': critic_loss.item(),  # Returns the critic loss as part of the results
            'overall_loss': overall_loss.item()  # Returns the overall loss as part of the results
        }

class GaussianNoise:
    """Generates Gaussian noise."""

    def __init__(self, mu=0.0, sigma=0.1):
        """
        :param mu: Mean of the Gaussian distribution.
        :param sigma: Standard deviation of the Gaussian distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def generate(self, shape):
        """
        Generate Gaussian noise based on a shape.

        :param shape: Shape of the noise to generate, typically the action's shape.
        :return: Numpy array with Gaussian noise.
        """
        noise = np.random.normal(self.mu, self.sigma, shape)
        return noise
