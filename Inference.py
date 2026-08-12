from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from belief_agent import Belief_agent
from belief_API import Belief_wrapper
from belief_env import Belief_env
from Neural_Network.local_map_encoder import ConditionalUnet1DWithLocalMap, ConditionalUnet1D
from common.map_utils import create_local_map
from Model.diffusion import Diffusion
from Model.flowmatching import FlowMatching




RRT_SRC = Path(__file__).resolve().parent / "external_repos" / "mapf_with_edge_bundles" / "src"
if str(RRT_SRC) not in sys.path:
    sys.path.append(str(RRT_SRC))

from printer import RRTPrinter
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

        self.goal_conditioning_bias = loaded_config.get("goal_conditioning_bias", 0)


        self.env = Belief_env()
        self.agent = Belief_agent(self.env)
        self.api = Belief_wrapper(self.env, self.agent)
        self.seed = loaded_config.get("seed", 5)
        goal = loaded_config.get("goal")
        if goal is None:
            goal = self.env.generate_random_goal(np.random.default_rng(self.seed))
        else:
            self.env.goal = np.asarray(goal, dtype=np.float64)
        self.goal = self.env.goal

        self.global_map = self.env.build_belief_global_map()
        self.map_center = (
            (self.env.x_bounds[1] - self.env.x_bounds[0]) / 2.0,
            (self.env.y_bounds[1] - self.env.y_bounds[0]) / 2.0,
        )

        super().__init__(
            start=self.env.env_start,
            goal=self.goal,
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
            udf_seed=self.seed,
            debug_flag=False,
            print_logs=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        unet_base_channels = {
            "small": 64,
            "medium": 256,
            "large": 512,
            "xlarge": 1024,
        }

        unet_size = loaded_config.get("unet_size", "large")
        unet_down_dims = loaded_config.get("unet_down_dims", [1, 2, 4])

        obs_dim = self.agent.state_length
        action_dim = self.agent.action_length
        input_channels = action_dim if self.prediction_type == "actions" else obs_dim
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
                input_channels=input_channels,
                output_channels=input_channels,
                encoder_name=self.local_map_encoder,
                embedding_dim=embedding_dim,
                additional_global_cond_dim=global_cond_dim,
                local_map_size=self.local_map_size,
                base_channels=unet_base_channels[unet_size],
                down_dims=unet_down_dims,
            )
        else:
            noise_pred_net = ConditionalUnet1D(
                input_channels=input_channels,
                output_channels=input_channels,
                global_cond_dim=global_cond_dim,
                base_channels=unet_base_channels[unet_size],
                down_dims=unet_down_dims,
            )

        checkpoint = loaded_config.get("checkpoint",  "beliefmaze_28_05_23_52_epoch_4_step_2000.pt")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / checkpoint_path
        checkpoint_state = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint_state.get("model_state_dict", checkpoint_state.get("noise_pred_net_state_dict"))
        if state_dict is None:
            raise KeyError("Checkpoint is missing model_state_dict/noise_pred_net_state_dict")
        noise_pred_net.load_state_dict(state_dict)
        noise_pred_net = noise_pred_net.to(device).eval()

        sampler_kwargs = {
            "env_id": "beliefmaze",
            "pred_horizon": self.pred_horizon,
            "action_dim": self.agent.action_length,
            "prediction_type": self.prediction_type,
            "obs_history": self.obs_history,
            "action_history": self.action_history,
            "position_conditioned": self.position_conditioned,
            "goal_conditioned": self.goal_conditioned,
            "local_map_conditioned": self.local_map_conditioned,
            "local_map_size": self.local_map_size,
            "device": device,
        }
        if self.policy == "diffusion":
            initial_model = Diffusion(
                noise_pred_net,
                num_diffusion_iters=self.diffusion_iters,
                **sampler_kwargs,
            ).eval()
        elif self.policy == "flow_matching":
            initial_model = FlowMatching(
                noise_pred_net,
                ode_steps=self.diffusion_iters,
                **sampler_kwargs,
            ).eval()
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        self.model = initial_model.to(device).eval()

    def sample_random_point(self):
        r = self.rng.random()
        if r < self.goal_conditioning_bias:
            random_point = self.goal
        else:
            # random_point = self.get_random_point(self.env, self.agent, self.rng)
            random_point = self.get_random_point(self.env, self.env.static_circular_obstacles,
                                                 self.env.static_rectangular_obstacles, self.rng)
        if self.debug_flag:
            print("Sampled random point: ", random_point)
        return random_point
    
    #overwrites RRT function with sampling actions from diffusion/flow-matching model
    def _select_best_extension_candidate(self, parent_node_id, parent_node, random_point):
        random_time = self.get_time()

        ids, states, controls, timesteps = self.get_path_to_node_id(parent_node_id)
        action_seq = None

        raw_obs_seq = np.asarray(states, dtype=np.float32)
        if self.position_conditioned:
            obs_seq = raw_obs_seq[-self.obs_history:][None, :, :]
        else:
            obs_seq = np.zeros((1,self.obs_history,2))

        if len(controls)>0:
            raw_action_seq = np.asarray(controls, dtype=np.float32)
            action_seq = raw_action_seq[-self.action_history:][None, :, :]

        x,y = parent_node.state[:2]
        local_map = create_local_map(self.global_map,x,y,theta=0.0,map_size=self.local_map_size,scale=self.local_map_scale,s_global=1.0,map_center = self.map_center)

        target = np.asarray([random_point], dtype=np.float32)
    
        actions = self.model(obs_seq,action_seq,target,local_map)

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

        last_action = mpc_actions[-1,:] #temporary ->Not logging all the actions but doesn't affect actual output

        best_candidate = (
            new_state,
            path_to_new_state,
            last_action,
            random_time
        )

        return best_candidate

    def run_guidance(self, pos): #state is pos
        state = torch.from_numpy(pos)
        state.requires_grad_()

        loss = self.inference_loss(state[:,:2])
        gradient = torch.autograd.grad(loss, state)[0]
        grad_norm = torch.linalg.vector_norm(gradient, dim = -1, keepdim = True) + 1e-8
        learning_rate = 1e-3

        state_new = state - learning_rate*(gradient/grad_norm)
        return state_new.detach().numpy()
        
    def inference_loss(self,state):
        x0,x1,y0,y1 = self.env.measurement_region
        waypoint = torch.tensor([(x0+x1)/2, (y0+y1)/2])
        waypoint = waypoint.repeat(len(state),1)
        waypoint_dist = torch.linalg.vector_norm(waypoint-state, dim = 1)

        #option 1
        #loss = torch.min(waypoint_dist**2)

        #option 2
        tau = 2
        loss = -tau * torch.log(torch.sum(torch.exp(-waypoint_dist/tau)))

        return loss
        
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
            obs = self.model(obs_cond,action_cond,self.goal, local_map)[0]
            idx, res = self.env.path_safe(obs)

            if not res:
                obs = obs[:idx,:] #Take part of state vector that is valid
            
            num_state_pred.append(len(obs))
            mpc_obs = obs[:self.action_horizon,:]


            #check if we hit the goal, either in the final state or anywhere in the path
            final_pos = mpc_obs[-1,:]
            reached,dist = self.api.reached_goal_function(final_pos,self.goal,self.env.goal_radius,self.agent)
            
            if reached:
                real_path = np.vstack((real_path,mpc_obs))        #add new rows to the overall path
                break
            else:
                if dist<threshold:
                    for i, pos in enumerate(mpc_obs,1):
                        ans,distance = self.api.reached_goal_function(pos,self.goal,self.env.goal_radius,self.agent)
                        if ans:
                            mpc_obs = mpc_obs[:i,:]
                            real_path = np.vstack((real_path,mpc_obs))
                            break

            real_path = np.vstack((real_path,mpc_obs))

            obs_cond = real_path[-obs_cond_len:,:]
            x,y = final_pos
            local_map = create_local_map(self.global_map,x,y,theta=0.0,map_size=self.local_map_size,scale=self.local_map_scale,s_global=1.0,map_center = self.map_center)

            num_iter+=1



            
def run_inference():
    name = "beliefmaze"
    infra = unique_inference(name)

    infra.run_action()

    ids, states, controls, timesteps = infra.get_path()

    output_dir = Path(__file__).resolve().parent / "media"
    output_dir.mkdir(exist_ok=True)
    v = RRTPrinter(infra.env, infra, ids)
    v.print_rrt(str(output_dir / "Pranith_flow_path.png"))


if __name__ == "__main__":
    run_inference()



                        





            






    
    
    
