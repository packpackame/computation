import torch
from torch import nn
from functools import partial
import numpy as np
from tqdm import tqdm
from .utils import (
    make_beta_schedule,
    extract,
    default,
)

from .unet import Unet

DEFAULT_BETA_SCHEDULE = {
        'schedule': 'linear',
        'linear_end': 0.99999999999999,
        'linear_start': 1e-3,
        'n_timestep': 10
    }


class Reconstructor(nn.Module):
    def __init__(self, checkpoint, beta_schedule=None):
        super(Reconstructor, self).__init__()
        if beta_schedule is None:
            beta_schedule = DEFAULT_BETA_SCHEDULE
        model_state = torch.load(checkpoint)
        self.params = model_state['hyper_parameters']
        model_design = model_state['hyper_parameters']['generator']['model_hyperparams']
        generator = Unet(**model_design)
        self.generator = generator
        self.load_state_dict(model_state['state_dict'])
        self.beta_schedule = beta_schedule
        self.set_new_noise_schedule()
        self.num_timesteps = self.beta_schedule['n_timestep']
        self.eval()

    def forward(self, x, gammas):
        return self.generator(x, gammas)

    def set_new_noise_schedule(self, device=torch.device("cuda")):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule)
        betas = (
            betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        )
        alphas = 1.0 - betas

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1.0, gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("gammas", to_torch(gammas))
        self.register_buffer("sqrt_recip_gammas", to_torch(np.sqrt(1.0 / gammas)))
        self.register_buffer("sqrt_recipm1_gammas", to_torch(np.sqrt(1.0 / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - gammas_prev) / (1.0 - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(gammas_prev) / (1.0 - gammas)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - gammas_prev) * np.sqrt(alphas) / (1.0 - gammas)),
        )

    def predict_start_from_noise(self, y_t, t, noise):
        return (
                extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t
                - extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat
                + extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, y_t.shape
        )
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
            y_t, t=t, noise=self.forward(torch.cat([y_cond, y_t], dim=1), noise_level)
        )

        if clip_denoised:
            y_0_hat.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t
        )
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond
        )
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, sample_num=2):
        b, *_ = y_cond.shape
        assert (
                self.num_timesteps > sample_num
        ), "num_timesteps must greater than sample_num"
        sample_inter = self.num_timesteps // sample_num
        y_t = default(
            y_t,
            lambda: torch.randn((y_cond.shape[0], 3, y_cond.shape[2], y_cond.shape[3])),
        ).to(
            y_cond.device
        )  # torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step",
                total=self.num_timesteps,
        ):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr