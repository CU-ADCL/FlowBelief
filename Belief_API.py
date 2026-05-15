from pathlib import Path
import sys

import numpy as np

from belief_agent import Belief_agent
from belief_env import Belief_env



RRT_SRC = Path(__file__).resolve().parent / "external_repos" / "mapf_with_edge_bundles" / "src"
if str(RRT_SRC) not in sys.path:
    sys.path.append(str(RRT_SRC))

from Environments import RectangleObstacle2D
from rrt import RRT
from printer import RRTPrinter


class Belief_wrapper():
    def __init__(self,belief_env,belief_agent):
        self.env = belief_env
        self.agent = belief_agent


    def random_point_function(self,env,static_circular_obstacles,static_rectangular_obstacles,rng):
        x = rng.uniform(env.x_bounds[0],env.x_bounds[1])
        y = rng.uniform(env.y_bounds[0],env.y_bounds[1])

        return np.array([x,y], dtype=np.float64)

    def isvalid_function(self, path_to_new_state, agent_radius, env_size, circ_obs,rec_obs,
        dyn_obs, obstacle_buffer, boundary_buffer, start_time, time_duration, dt_per_step):
        
        return self.env.path_safe(path_to_new_state)

    #cost is just distance
    def cost_function(self,env,agent,parent_state,action,duration, path):
        states = len(path)-1
        tot_dist = 0

        for i in range(states):
            s1 = path[i,:2]
            s2 = path[i+1,:2]
            tot_dist += np.linalg.norm(s2-s1)

        return float(tot_dist)

    def reached_goal_function(self,state, goal, goal_radius, agent):
        dist_to_goal = np.linalg.norm(goal[:2] - state[:2])

        return dist_to_goal<=goal_radius, dist_to_goal


#Run RRT
env = Belief_env()
agent = Belief_agent(env)
api = Belief_wrapper(env, agent)

env.obstacles = [
    RectangleObstacle2D(
        x=(x1 + x2) / 2.0,
        y=(y1 + y2) / 2.0,
        w=(x2 - x1),
        h=(y2 - y1),
    )
    for (x1, x2, y1, y2) in env.obstacle_region
]

numpath = 1
path_list = []
seed_start = 75
output_dir = Path("media")
output_dir.mkdir(exist_ok=True)

for i in range(numpath):
    seed = seed_start + i

    rrt = RRT(
        start=env.env_start,
        goal=env.goal,
        goal_radius=env.goal_radius,
        env=env,
        agent=agent,
        use_fixed_sampling_time=True,
        sampling_time_step=1.0,
        minimum_time_step=0.5,
        max_iter=1000,
        planning_time=10.0,
        isvalid_function=api.isvalid_function,
        cost_function=api.cost_function,
        random_point_function=api.random_point_function,
        reached_goal_function=api.reached_goal_function,
        udf_seed=seed,
        debug_flag=False,
        print_logs=True,
    )

    rrt.plan_path()

    ids, states, controls, timesteps = rrt.get_path()

    result = {
        "seed": seed,
        "path_found": rrt.path_found,
        "path_cost": rrt.path_cost,
        "path_time": rrt.path_time,
        "ids": ids,
        "states": states,
        "controls": controls,
        "timesteps": timesteps,
    }
    path_list.append(result)
    v = RRTPrinter(env, rrt,ids)
    v.print_rrt(str(output_dir / "Pranith_path.png"))

successful_paths = [result for result in path_list if result["path_found"]]
print("Total runs:", len(path_list))
print("Successful paths:", len(successful_paths))
