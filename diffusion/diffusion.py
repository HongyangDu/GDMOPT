# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import additional helper functions and utils
from .helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses
)
from .utils import Progress, Silent

# Define the main Diffusion class that inherits from PyTorch's nn.Module
class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='vp', n_timesteps=5,
                 loss_type='l2', clip_denoised=True, bc_coef=False):
        # Call parent constructor
        super(Diffusion, self).__init__()

        # Set initial attributes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model

        # Define the diffusion beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        # Define alpha parameters related to the beta schedule
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.bc_coef = bc_coef

        # Register these values as buffers in the module, which PyTorch will track
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Pre-calculate some quantities for the diffusion process and posterior
        # distribution calculation based on alpha and beta schedules
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # More pre-calculations for the posterior distribution
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # Log calculation clipped to avoid log(0)
        # ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # Select the appropriate loss function from the predefined Losses dictionary
        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#
    # Section to define the sampling methods for the diffusion
    # Predict the original state given the diffused state at time t and noise
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.explore_solution, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.bc_coef:
            return noise
        else:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )


    # Define the mean, variance, and log variance of the posterior distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # Define the mean and variance of the prior distribution
    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    # Sample from the prior distribution
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        # with torch.random.fork_rng():
        #     torch.manual_seed(t)
        #     noise = torch.randn_like(x)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        # with torch.random.fork_rng():
        #     torch.manual_seed(0)
        #     x = torch.randn(shape, device=device)
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
            # max_action = 1.0
            # ====== for inference ======
            # x.clamp_(-self.max_action, self.max_action)
            # actions = torch.abs(x)
            # Aution = actions.detach().numpy()
            # normalized_weights = Aution / np.sum(Aution)
            # total_power = 12
            # actf = normalized_weights * total_power
            # actff = torch.from_numpy(actf).float()
            # print('x', actff)
            # ===========================

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    # Generate a sample by using the p_sample_loop method and clamp the values within the max action range
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        # Clamping the actions to be between -max_action and max_action
        return action.clamp_(-self.max_action, self.max_action)
        # return action

    # ------------------------------------------ training ------------------------------------------#
    # Define the sampling method for the posterior distribution
    def q_sample(self, x_start, t, noise=None):
        # if noise is not provided, generate random noise
        if noise is None:
            noise = torch.randn_like(x_start)
        # compute the diffused state
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    # Compute the losses based on the predictions from the model
    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        # compute the noisy state
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict the noise or the original state based on the noisy state
        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.bc_coef:
            loss = self.loss_fn(x_recon, x_start, weights)
            # else compute loss based on the predicted original state and the actual original state
        else:
            loss = self.loss_fn(x_recon, noise, weights)
        # loss = self.loss_fn(x_recon, noise, weights)
        return loss

    # Compute the total loss by sampling different timesteps for each data in the batch
    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        # sample a different timestep for each data in the batch
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    # Generate a sample from the model
    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)