import math
import numpy as np


class Belief_agent():

    state_length = 2
    action_length = 2
    measurement_size = 2
    
    def __init__(self,env=None):
        covariance_init = np.eye(self.state_length)

        self.env = env
        self.state = [self.env.env_start,covariance_init.flatten()]
        self.input = np.zeros((self.action_length))

        self.A = np.eye(self.state_length)
        self.B = np.eye(self.action_length)
        self.C = np.eye(self.measurement_size) #assuming measurementsize=statesize
        self.Q = np.eye(self.state_length)

        self.radius = 0.0
        self.distance_metric_state_size = 2

        self.u_max = 5
        self.u_min = -5

    def step_state(self):
        pos_next = self.A @ self.state[0] + self.B @ self.input
        cov_next = self.step_covariance(pos_next)

        pos_next = self.env.clip_pos(pos_next)

        return [pos_next,cov_next]


    def step_covariance(self,pos_next):
        covmat = self.state[1].reshape(self.state_length, self.state_length)

        p_minus = (self.A @ covmat @ self.A.T) + self.Q

        S = (self.C @ p_minus @ self.C.T) + self.env.R(pos_next)
        K = p_minus @ self.C.T @ np.linalg.inv(S)

        p_plus = (np.eye(self.state_length) - (K@self.C)) @ p_minus
        
        return p_plus.flatten()

    def step_mult(self,steps):
        path = np.zeros((steps,self.state_length + (self.state_length**2)))

        for i in range(steps):
            pos,cov = self.step_state()
            self.state = [pos,cov]
            x,y = pos
            path[i,0] = x 
            path[i,1] = y
            path[i,2:6] = cov
        
        return [pos, cov], path

    
    #wrapper functions for state-space RRT code
    def get_random_action(self,rng):
        return rng.uniform(self.u_min, self.u_max, size=self.action_length)


    def get_next_state(self,state,action,dt,num_steps):

        og_state = self.state
        og_input = self.input

        self.state = [state.copy(), og_state[1].copy()] #for belief space, need to step with a passed in covariance too
        self.input = action.copy()

        new_state,path = self.step_mult(num_steps)
        new_state = new_state[0]            #only want position
        new_path = path[:,:2]

        self.state = og_state
        self.input = og_input

        return new_state, new_path



