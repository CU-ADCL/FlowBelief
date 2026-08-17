import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from Model.policy_base import PolicySamplerBase

class Diffusion(PolicySamplerBase):
    def __init__(self, model, num_diffusion_iters, **kwargs):

        super().__init__(model, num_diffusion_iters=num_diffusion_iters, **kwargs)
        self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_diffusion_iters,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )
        
    def forward(self, obs_seq, action_seq, goal=None, local_map=None):
        cond_vec, local_map = self._build_conditions(
            obs_seq,
            action_seq,
            goal,
            local_map,
        )
        batch_size = cond_vec.shape[0]

        with torch.inference_mode():
            sample = self._initial_sample(batch_size)
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for timestep in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    sample=sample,
                    local_map=local_map,
                    timestep=timestep,
                    global_cond=cond_vec,
                )
                sample = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=sample,
                ).prev_sample

        return self._denormalize_prediction(sample)

    def loss(self, obs_seq, action_seq, goal, local_map):
        obs_seq_np = self._to_numpy(obs_seq)
        action_seq_np = self._to_numpy(action_seq)

        cond_vec, local_map = self._build_conditions(
            obs_seq_np[:, :self.obs_history, :],
            action_seq_np[:, :self.action_history, :],
            goal,
            local_map,
            goal_is_preprocessed=True,
        )

        target = self._prepare_prediction_target(obs_seq_np, action_seq_np)
        batch_size = target.shape[0]
        noise = torch.randn(target.shape, device=self.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noisy_target = self.noise_scheduler.add_noise(target, noise, timesteps)
        noise_pred = self.model(noisy_target, local_map, timesteps, global_cond=cond_vec)
        
        return F.mse_loss(noise_pred, noise)
