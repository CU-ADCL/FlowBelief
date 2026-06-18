import math
import numpy as np


class Belief_agent():

    state_length = 2
    action_length = 2
    measurement_size = 2
    
    def __init__(self,env=None):
        covariance_init = np.eye(self.state_length)

        self.env = env
    
        self.A = np.eye(self.state_length)
        self.B = np.eye(self.action_length)
        self.C = np.eye(self.measurement_size) #assuming measurementsize=statesize
        self.Q = np.eye(self.state_length)

        self.radius = 0.0
        self.distance_metric_state_size = 2

        self.u_max = 5
        self.u_min = -5

    def step_state(self, state, action):
        pos = state[:self.state_length]
        cov = state[self.state_length:]

        if len(cov) == 0: #if only doing state-space planning
            cov = np.zeros((self.state_length**2))

        pos_next = self.A @ pos + self.B @ action
        cov_next = self.step_covariance(pos_next,cov)

        pos_next = self.env.clip_pos(pos_next)

        new_state = np.append(pos_next, cov_next)

        return new_state


    def step_covariance(self, pos_next, cov):
        covmat = cov.reshape(self.state_length, self.state_length)

        p_minus = (self.A @ covmat @ self.A.T) + self.Q

        S = (self.C @ p_minus @ self.C.T) + self.env.R(pos_next)
        K = p_minus @ self.C.T @ np.linalg.inv(S)

        p_plus = (np.eye(self.state_length) - (K@self.C)) @ p_minus
        
        return p_plus.flatten()

    def step_mult(self, steps, state, actions):
        path = np.zeros((steps,self.state_length + (self.state_length**2)))
        flag = 0

        if len(actions) == 1: #Means this is for RRT training data, NOT model inference
            flag = 1    #if there is only 1 action, and then therefore usually more steps then actions

        for i in range(steps):
            if not flag:
                state_next = self.step_state(state, actions[i,:])
            else: 
                state_next = self.step_state(state, actions[0,:])
        
            path[i,:] = state_next
            state=state_next
        
        return path[-1,:], path

    
    #wrapper functions for state-space RRT code
    def get_random_action(self, rng):
        return rng.uniform(self.u_min, self.u_max, size=(1,self.action_length))


    def get_next_state(self, state, actions, dt, num_steps):

        new_state, path = self.step_mult(num_steps, state, actions)

        #only extract position
        new_state = new_state[:self.state_length]
        new_path = path[:,:self.state_length]

        return new_state, new_path

