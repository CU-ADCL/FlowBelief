import math
import numpy as np

class Belief_env():

#set up environment parameters
    def __init__(self):

        self.measurement_size = 2  #need to change across gym and agent files

        self.x_bounds = (0, 100)
        self.y_bounds = (0, 100)

        self.dt = 0.05

        self.goal = np.array([90,90]);
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
    def check_goal(self,pos):
        dist = np.linalg.norm(self.goal-pos)  #return distance to goal

        return dist<=self.goal_radius

    #also checks out of bounds even though state gets clipped in dynamics fcn
    def check_collision(self, pos):
        x,y = pos
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds


        for i in range(len(self.obstacle_region)):
            if (self.obstacle_region[i][0] <=x<= self.obstacle_region[i][1]
             and self.obstacle_region[i][2]<=y<= self.obstacle_region[i][3]):
                return True

        if xmin<=x<xmax and ymin<=y<ymax:
            return False
        else: 
            return True

    def path_safe(self, path):
        for pos in path:
            if self.check_collision(pos):
                return False
        return True

    #might want to add covariance clipping if that exists
    def clip_pos(self,pos):
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds

        return np.array([np.clip(pos[0], xmin,xmax), np.clip(pos[1],ymin,ymax)])
    