from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from belief_agent import Belief_agent
from belief_API import Belief_wrapper
from belief_env import Belief_env
from local_map_encoder import ConditionalUnet1DWithLocalMap, ConditionalUnet1D
from policies.fm_policy import DiffusionSampler
from common.map_utils import create_local_map




RRT_SRC = Path(__file__).resolve().parent / "external_repos" / "mapf_with_edge_bundles" / "src"
if str(RRT_SRC) not in sys.path:
    sys.path.append(str(RRT_SRC))

from rrt import RRT

class unique_inference(RRT):
    def __init__(self, cfg_file="beliefmaze", config_path=None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "cfgs" / f"{cfg_file}.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path, "r") as file:
            loaded_config = yaml.safe_load(file) or {}

        self.config_path = config_path
        self.loaded_config = loaded_config

        self.prediction_type = loaded_config.get("prediction_type", "actions")
        self.position_conditioned = loaded_config.get("position_conditioned", True)
        self.goal_conditioned = loaded_config.get("goal_conditioned", True)
        self.local_map_conditioned = loaded_config.get("local_map_conditioned", True)
        self.obs_history = loaded_config.get("obs_history", 1)
        self.action_history = loaded_config.get("action_history", 1)

        self.policy = loaded_config.get("policy", "flow_matching")
        self.model_predictive_control = loaded_config.get("model_predictive_control", False)

        self.pred_horizon = loaded_config.get("pred_horizon", 10)
        self.action_horizon = loaded_config.get("action_horizon", 2)

        self.local_map_size = loaded_config.get("local_map_size", 20)
        self.local_map_scale = loaded_config.get("local_map_scale", 0.2)
        self.local_map_embedding_dim = loaded_config.get("local_map_embedding_dim", 400)
        self.local_map_encoder = loaded_config.get("local_map_encoder", "resnet")

        self.diffusion_iters = loaded_config.get("num_diffusion_iters", 1)

        self.env = Belief_env()
        self.agent = Belief_agent(self.env)
        self.api = Belief_wrapper(self.env, self.agent)

        self.global_map = self.env.build_belief_global_map()
        self.map_center = self.env.y_bounds

        super().__init__(
            start=self.env.env_start,
            goal=self.env.goal,
            goal_radius=self.env.goal_radius,
            env=self.env,
            agent=self.agent,
            use_fixed_sampling_time=True,
            sampling_time_step=1.0,
            minimum_time_step=0.5,
            max_iter=1000,
            planning_time=10.0,
            isvalid_function=self.api.isvalid_function,
            cost_function=self.api.cost_function,
            random_point_function=self.api.random_point_function,
            reached_goal_function=self.api.reached_goal_function,
            udf_seed=loaded_config.get("seed", 42),
            debug_flag=False,
            print_logs=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        unet_dims = {
            "small": [64, 128, 256],
            "medium": [256, 512, 1024],
            "large": [512, 1024, 2048],
            "xlarge": [1024, 2048, 4096],
        }

        unet_size = loaded_config.get("unet_size", "large")

        obs_dim = self.agent.state_length
        action_dim = self.agent.action_length
        input_dim = action_dim if self.prediction_type == "actions" else obs_dim
        global_cond_dim = (
            obs_dim * self.obs_history
            + 2 * self.goal_conditioned
            + self.action_history * action_dim
        )

        if self.local_map_conditioned:
            embedding_dim = (
                self.local_map_size ** 2
                if self.local_map_encoder.lower() in ("identity", "mlp")
                else self.local_map_embedding_dim
            )
            noise_pred_net = ConditionalUnet1DWithLocalMap(
                input_dim=input_dim,
                encoder_name=self.local_map_encoder,
                embedding_dim=embedding_dim,
                additional_global_cond_dim=global_cond_dim,
                local_map_size=self.local_map_size,
                down_dims=unet_dims[unet_size],
            )
        else:
            noise_pred_net = ConditionalUnet1D(
                input_dim=input_dim,
                global_cond_dim=global_cond_dim,
                down_dims=unet_dims[unet_size],
            )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        checkpoint = loaded_config.get("checkpoint",  "beliefmaze_29_05_HH_MM_epoch_5_step_XXXX.pt")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / checkpoint_path
        checkpoint_state = torch.load(checkpoint_path, map_location=device)
        noise_pred_net.load_state_dict(checkpoint_state["noise_pred_net_state_dict"])
        noise_pred_net = noise_pred_net.to(device).eval()

        initial_model = DiffusionSampler(noise_pred_net, noise_scheduler, 'beliefmaze',
                                                      policy=self.policy,
                                                      pred_horizon=self.pred_horizon, action_dim=self.agent.action_length,
                                                      prediction_type=self.prediction_type,
                                                      obs_history=self.obs_history, action_history=self.action_history,
                                                      num_diffusion_iters=self.diffusion_iters, position_conditioned=self.position_conditioned, goal_conditioned=self.goal_conditioned, 
                                                      local_map_conditioned=self.local_map_conditioned, local_map_size=self.local_map_size).eval()

        self.model = initial_model.to(device).eval()

    #overwrites RRT function with sampling actions from diffusion/flow-matching model
    def _select_best_extension_candidate(self, parent_node_id, parent_node, random_point):
        random_time = self.get_time()

        ids, states, controls, timesteps = self.get_path_to_node_id(parent_node_id)
        action_seq = None

        raw_obs_seq = np.asarray([states])
        if self.position_conditioned:
            obs_seq = raw_obs_seq[-self.obs_history:,:]
        else:
            obs_seq = np.zeros((1,self.obs_history,2))

        if len(controls)>0:
            raw_action_seq = np.asarray([controls])
            action_seq = raw_action_seq[-self.action_history:,:]

        x,y = parent_node.state[:2]
        local_map = create_local_map(self.global_map,x,y,theta=0.0,map_size=self.local_map_size,scale=self.local_map_scale,s_global=1.0,map_center = self.map_center)

        actions = self.model(obs_seq,action_seq,self.env.goal,local_map) #how do I add local map?

        mpc_actions = actions[0,:self.action_horizon,:]

        new_state, path_to_new_state = self.agent.get_next_state(
                parent_node.state,
                mpc_actions,
                random_time,
                self.action_horizon
        )

        accept_new_node = self.isvalid(
            path_to_new_state,
            self.agent.radius,
            self.env.size,
            self.static_circular_obstacles,
            self.static_rectangular_obstacles,
            self.dynamic_agent_obstacles,
            self.env.obstacle_buffer,
            self.env.boundary_buffer,
            parent_node.time_elapsed,
            random_time,
            self.minimum_time_step
        )

        if not accept_new_node:
            return None
    
        best_candidate = (
            new_state,
            path_to_new_state,
            mpc_actions,
            random_time
        )

        return best_candidate
    
    #Run RRT+action model
    def run_action(self):
        self.plan_path()

    #Run prediction_type = "observations" diffusion/flow_matching model
    def run_obs(self):   
        obs_cond_len = self.obs_history
        action_cond_len = self.action_history

        pos_start = self.env.env_start
        x,y = pos_start

        obs_cond = np.zeros((obs_cond_len-1,2))
        obs_cond = np.vstack((obs_cond, pos_start)) #length of conditioning is matched
        action_cond = np.zeros((action_cond_len,self.agent.action_length)) #no actual action conditioning happening btw

        local_map = create_local_map(self.global_map,x,y,theta=0.0,map_size=self.local_map_size,scale=self.local_map_scale,s_global=1.0,map_center = self.map_center)

    
        max_iter = 1000
        num_iter = 0
        threshold = 3.0

        num_state_pred = []
        real_path = pos_start

        while num_iter <= max_iter:
            obs = self.model(obs_cond,action_cond,self.env.goal, local_map)[0]
            idx, res = self.env.path_safe(obs)

            if not res:
                obs = obs[:idx,:] #Take part of state vector that is valid
            
            num_state_pred.append(len(obs))
            mpc_obs = obs[:self.action_horizon,:]


            #check if we hit the goal, either in the final state or anywhere in the path
            final_pos = mpc_obs[-1,:]
            reached,dist = self.api.reached_goal_function(final_pos,self.env.goal,self.env.goal_radius,self.agent)
            
            if reached:
                real_path = np.vstack((real_path,mpc_obs))        #add new rows to the overall path
                break
            else:
                if dist<threshold:
                    for i, pos in enumerate(mpc_obs,1):
                        ans,distance = self.api.reached_goal_function(pos,self.env.goal,self.env.goal_radius,self.agent)
                        if ans:
                            mpc_obs = mpc_obs[:i,:]
                            real_path = np.vstack((real_path,mpc_obs))
                            break

            real_path = np.vstack((real_path,mpc_obs))

            obs_cond = real_path[-obs_cond_len:,:]
            x,y = final_pos
            local_map = create_local_map(self.global_map,x,y,theta=0.0,map_size=self.local_map_size,scale=self.local_map_scale,s_global=1.0,map_center = self.map_center)

            num_iter+=1



            

        




                        





            






    
    
    
