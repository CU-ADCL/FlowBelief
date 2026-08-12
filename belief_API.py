from pathlib import Path
import pickle
import sys

import numpy as np

from belief_agent import Belief_agent
from belief_env import Belief_env



RRT_SRC = Path(__file__).resolve().parent / "external_repos" / "mapf_with_edge_bundles" / "src"
if str(RRT_SRC) not in sys.path:
    sys.path.append(str(RRT_SRC))

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
        
        index, ans = self.env.path_safe(path_to_new_state)
        return ans
        
    #cost is just distance for now
    def cost_function(self,env,agent,parent_state,action,duration, path):
        return np.linalg.norm(path[-1,:2] - parent_state[:2])

    def reached_goal_function(self,state, goal, goal_radius, agent):
        dist_to_goal = np.linalg.norm(goal[:2] - state[:2])

        return dist_to_goal<=goal_radius, dist_to_goal

""" 
TODO->
Add eval code
 """
def collect_training_paths():
    #Run RRT
    env = Belief_env()
    agent = Belief_agent(env)
    api = Belief_wrapper(env, agent)

    numpath = 10000
    total_runs = 0
    dataset = []
    seed_start = 10
    output_dir = Path("media")
    output_dir.mkdir(exist_ok=True)
    dataset_dir = Path("datasets")
    dataset_dir.mkdir(exist_ok=True)
    dataset_path = dataset_dir / "belief_episodes.pkl"
    save_every_successes = 100

    for i in range(numpath):
        seed = seed_start + i
        rng = np.random.default_rng(seed)
        goal = env.generate_random_goal(rng)

        rrt = RRT(
            start=env.env_start,
            goal=goal,
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
            print_logs=False,
        )

        rrt.plan_path()
        total_runs += 1

        ids, states, controls, timesteps = rrt.get_path()

        if rrt.path_found:
            observations = np.asarray(states, dtype=np.float32)
            actions = np.asarray(controls, dtype=np.float32)

            if len(observations) >= 2 and len(actions) >= 1:
                episode = {
                    "observations": observations,
                    "actions": actions,
                    "goal": np.asarray(goal, dtype=np.float32),
                    "goal_radius": float(env.goal_radius),
                    "seed": seed,
                    "path_cost": rrt.path_cost,
                    "path_time": rrt.path_time,
                }
                dataset.append(episode)

            if len(dataset) == 1:
                v = RRTPrinter(env, rrt, ids)
                v.print_rrt(str(output_dir / "Pranith_path.png"))

            if len(dataset) > 0 and len(dataset) % save_every_successes == 0:
                with open(dataset_path, "wb") as f:
                    pickle.dump(dataset, f)
                print(f"Checkpoint saved: {len(dataset)} episodes after {total_runs} runs")

    print("Total runs:", total_runs)
    print("Saved successful episodes:", len(dataset))

    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
    print("Dataset path:", dataset_path)

if __name__ == "__main__":
    collect_training_paths()
