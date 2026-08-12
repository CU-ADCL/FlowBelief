import math
import numpy as np
from pathlib import Path
import sys

RRT_SRC = Path(__file__).resolve().parent / "external_repos" / "mapf_with_edge_bundles" / "src"
if str(RRT_SRC) not in sys.path:
    sys.path.append(str(RRT_SRC))

from Environments import RectangleObstacle2D

class Belief_env():

#set up environment parameters
    def __init__(self, goal=None, rng=None):
        self.measurement_size = 2  #need to change across gym and agent files

        self.x_bounds = (0, 100)
        self.y_bounds = (0, 100)

        self.dt = 0.05

        self.goal_radius = 10
        self.obstacle_region = [(0,43,50,80), (50,100,50,80)]
        self.measurement_region = (75,100,0,30)

        self.env_start = np.array([1.0, 1.0], dtype=np.float64)

        #wrapper fields for RRT
        self.size = np.array([100.0, 100.0], dtype=np.float64)

        self.boundary_buffer = 0.0
        self.obstacle_buffer = 0.0

        self.static_circular_obstacles = np.empty((0, 3), dtype=np.float64)
        self.static_rectangular_obstacles = np.array(self.obstacle_region, dtype=np.float64)

        self.obstacles = [
            RectangleObstacle2D(
                x=(x1 + x2) / 2.0,
                y=(y1 + y2) / 2.0,
                w=(x2 - x1),
                h=(y2 - y1),
            )
            for (x1, x2, y1, y2) in self.obstacle_region
        ]

        self.goal = None
        if goal is not None:
            self.goal = np.asarray(goal, dtype=np.float64)
        elif rng is not None:
            self.generate_random_goal(rng)

    def generate_random_goal(self, rng=None, min_distance_from_start=None, max_attempts=10000):
        if rng is None:
            rng = np.random.default_rng()

        if min_distance_from_start is None:
            min_distance_from_start = self.goal_radius

        for _ in range(max_attempts):
            goal = np.array(
                [
                    rng.uniform(self.x_bounds[0], self.x_bounds[1]),
                    rng.uniform(self.y_bounds[0], self.y_bounds[1]),
                ],
                dtype=np.float64,
            )

            if self.check_collision(goal):
                continue
            if np.linalg.norm(goal - self.env_start) < min_distance_from_start:
                continue
            self.goal = goal
            return self.goal

        raise RuntimeError("Could not sample a valid goal inside the environment.")

    def build_belief_global_map(self, s_global=1.0):
        width = int((self.x_bounds[1] - self.x_bounds[0]) / s_global)
        height = int((self.y_bounds[1] - self.y_bounds[0]) / s_global)

        global_map = np.zeros((height, width), dtype=np.float32)

        for x1, x2, y1, y2 in self.obstacle_region:
            col1 = int(x1 / s_global)
            col2 = int(x2 / s_global)
            row1 = height - int(y2 / s_global)
            row2 = height - int(y1 / s_global)
            global_map[row1:row2, col1:col2] = 1.0

        return global_map

    def cond_measurement_region(self,pos):
        x,y = pos
        x1,x2,y1,y2 = self.measurement_region

        if x1<=x<=x2 and y1<=y<=y2:
            return True
        else:
            return False


    def R(self,pos):
        if self.cond_measurement_region(pos):
            return np.eye(self.measurement_size) * 0.1
        else:
            return np.eye(self.measurement_size) * 50


    #Following check_goal and check_collision is for regular state-space RRT 
    def check_goal(self, pos, goal=None, goal_radius=None):
        if goal is None:
            if self.goal is None:
                raise ValueError("check_goal requires a goal when the environment has no current goal.")
            goal = self.goal
        if goal_radius is None:
            goal_radius = self.goal_radius

        dist = np.linalg.norm(np.asarray(goal)[:2] - np.asarray(pos)[:2])  #return distance to goal

        return dist<=goal_radius

    #also checks out of bounds even though state gets clipped in dynamics fcn
    def check_collision(self, pos):
        x,y = pos
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds


        for i in range(len(self.obstacle_region)):
            if (self.obstacle_region[i][0] <=x<= self.obstacle_region[i][1]
             and self.obstacle_region[i][2]<=y<= self.obstacle_region[i][3]):
                return True

        if xmin<x<xmax and ymin<y<ymax:
            return False
        else: 
            return True

    def path_safe(self, path):
        for i, pos in enumerate(path):
            if self.check_collision(pos):
                return i, False
        return len(path), True

    #might want to add covariance clipping if that exists
    def clip_pos(self,pos):
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds

        return np.array([np.clip(pos[0], xmin,xmax), np.clip(pos[1],ymin,ymax)])
    
