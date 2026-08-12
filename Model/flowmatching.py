import torch
import torch.nn.functional as F

from common.fm_utils import get_timesteps
from Model.policy_base import PolicySamplerBase


class FlowMatching(PolicySamplerBase):
    def __init__(self, model, ode_steps, **kwargs):

        super().__init__(model, num_diffusion_iters=ode_steps, **kwargs)
        self.ode_steps = ode_steps

    def forward(self, obs_hist, action_hist, goal=None, local_map=None):
        cond_vec, local_map = self._build_conditions(
            obs_hist,
            action_hist,
            goal,
            local_map,
        )
        batch_size = cond_vec.shape[0]

        with torch.no_grad():
            sample = self._initial_sample(batch_size)
            t0, dt = get_timesteps("exp", self.ode_steps, exp_scale=4.0)
            t0 = t0.to(self.device)
            dt = dt.to(self.device)

            for step in range(self.ode_steps):
                timesteps = torch.ones((batch_size), device=self.device) * t0[step]
                timesteps *= 20
                vel_pred = self.model(
                    sample=sample,
                    local_map=local_map,
                    timestep=timesteps,
                    global_cond=cond_vec,
                )
                sample = sample.detach().clone() + vel_pred * dt[step]

        return self._denormalize_prediction(sample)
    
    def loss(self, obs_seq, action_seq, goal, local_map):
        obs_seq_np = self._to_numpy(obs_seq)
        action_seq_np = self._to_numpy(action_seq)

        cond_vec, local_map = self._build_conditions(
            obs_seq_np[:,:self.obs_history,:],
            action_seq_np[:, :self.action_history, :],
            goal,
            local_map,
            goal_is_preprocessed=True,
        )

        target = self._prepare_prediction_target(obs_seq_np, action_seq_np)
        batch_size = target.shape[0]

        noise = torch.randn(target.shape, device=self.device)
        t = torch.rand((batch_size, 1, 1), device=self.device)
        z_t = t * target + (1.0 - t) * noise
        target_vel = target - noise
        timesteps = t.squeeze() * 20
        pred_vel = self.model(z_t, local_map, timesteps, global_cond=cond_vec)

        return F.mse_loss(pred_vel, target_vel)
