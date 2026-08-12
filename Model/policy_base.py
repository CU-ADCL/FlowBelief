import os

import numpy as np
import torch
from torch import nn


class PolicySamplerBase(nn.Module):
    def __init__(
            self,
            model,
            env_id,
            pred_horizon,
            action_dim,
            prediction_type="actions",
            obs_history=1,
            action_history=1,
            num_diffusion_iters=100,
            position_conditioned=False,
            goal_conditioned=True,
            local_map_conditioned=True,
            local_map_size=16,
            device=None,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        metadata_path = f"metadata/{env_id}.pt"
        if os.path.exists(metadata_path):
            self.metadata = torch.load(metadata_path, weights_only=False)
        else:
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        self.action_dim = action_dim
        self.prediction_type = prediction_type
        self.env_id = env_id
        self.pred_horizon = pred_horizon
        self.obs_history = obs_history
        self.action_history = action_history
        self.num_diffusion_iters = num_diffusion_iters
        self.goal_conditioned = goal_conditioned
        self.local_map_conditioned = local_map_conditioned
        self.local_map_size = local_map_size
        self.model = model

    def _to_numpy(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _normalize_observations(self, obs_hist):
        return (obs_hist - self.metadata["Observations_mean"]) / self.metadata["Observations_std"]

    def _normalize_actions(self, action_hist):
        return (action_hist - self.metadata["Actions_mean"]) / self.metadata["Actions_std"]

    def _denormalize_observations(self, obs_hist):
        return obs_hist * self.metadata["Observations_std"] + self.metadata["Observations_mean"]

    def _denormalize_actions(self, action_hist):
        return action_hist * self.metadata["Actions_std"] + self.metadata["Actions_mean"]

    def _denormalize_prediction(self, prediction):
            if self.prediction_type == "actions":
                prediction = prediction.detach().to("cpu").numpy()
                return self._denormalize_actions(prediction)
    
            if self.prediction_type == "observations":
                prediction = prediction.detach().to("cpu").numpy()
                return self._denormalize_observations(prediction)
    
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
    
    def _prepare_prediction_target(self, obs_hist, action_hist):
        if self.prediction_type == "actions":
            target = action_hist[:, -self.pred_horizon:, :]
            return torch.as_tensor(
                self._normalize_actions(target),
                device=self.device,
                dtype=torch.float32,
            )

        if self.prediction_type == "observations":
            target = obs_hist[:, -self.pred_horizon:, :]
            return torch.as_tensor(
                self._normalize_observations(target),
                device=self.device,
                dtype=torch.float32,
            )

        raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def _build_conditions(
            self,
            obs_hist,
            action_hist,
            goal,
            local_map,
            goal_is_preprocessed=False,
    ):

        obs_hist = self._to_numpy(obs_hist).copy()
        action_hist = self._to_numpy(action_hist)
     
        current_position = obs_hist[:, -1, :2]

        #Condition observations
        obs_hist = self._normalize_observations(obs_hist)
        cond_vec = torch.from_numpy(obs_hist)
        cond_vec = cond_vec.flatten(start_dim=1).to(self.device, dtype=torch.float32)

        #Condition actions
        if self.action_history > 0:
            action_hist = self._normalize_actions(action_hist)
            action_cond = torch.from_numpy(action_hist)
            action_cond = action_cond.flatten(start_dim=1).to(self.device, dtype=torch.float32)
            cond_vec = torch.cat([cond_vec, action_cond], dim=1)

        #Condition goal
        if self.goal_conditioned:
            goal = self._to_numpy(goal)
            if goal.ndim == 1:
                goal = np.expand_dims(goal, axis=0)
            if goal_is_preprocessed:
                relative_goal = goal
            else:
                relative_goal = np.tanh((goal - current_position) / self.local_map_size)
            relative_goal = torch.tensor(relative_goal, device=self.device, dtype=torch.float32)
            cond_vec = torch.cat([cond_vec, relative_goal], dim=1)

        #Condition local map
        if self.local_map_conditioned:
            if isinstance(local_map, np.ndarray):
                local_map = torch.from_numpy(local_map)
            local_map = local_map.to(self.device, dtype=torch.float32)
            if len(local_map.shape) == 2:
                local_map = local_map.unsqueeze(0)
            local_map = local_map * 2 - 1

        return cond_vec, local_map

    def _initial_sample(self, batch_size):
        if self.prediction_type == "actions":
            return torch.randn((batch_size, self.pred_horizon, self.action_dim), device=self.device)
        if self.prediction_type == "observations":
            obs_dim = len(self.metadata["Observations_mean"])
            return torch.randn((batch_size, self.pred_horizon, obs_dim), device=self.device)
        raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
