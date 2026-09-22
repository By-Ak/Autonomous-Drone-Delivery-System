import gymnasium as gym
import numpy as np

class Cityenv(gym.Env):   # Defining how our environment will look like 
    def __init__(self):
        super().__init__()
        self.size=7            # Initializing map size

        # Defining number of actions that will be used
        self.action_space=gym.spaces.Discrete(4)   

        # Defines what the agent can observe: the 7x7 city map and remaining battery percentage
        #self.observation_space=gym.spaces.Dict({"map":gym.spaces.Box(low=0,high=5,shape=(7,7)), 
        #                                       "battery":gym.spaces.Box(low=0,high=100,shape=(1,))})  
        # for now let this be in a comment, when we will implement battery, then the above command will be used

        self.observation_space=gym.spaces.Box(low=0,high=5,shape=(7,7))



    #defining how our self.observation will look like after reset or at start of an episode
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.battery=100

        self.drone_pos=[0,1]
        self.goal_pos=[5,4]
        self.restaurant_pos=[0,0]
        self.obstacles_pos=[[2,2],[3,1],[1,2]]
        self.charging_station_pos=[[4,1],[6,3]]

        self.observation=np.zeros((7,7),dtype=np.int32)

        self.observation[self.drone_pos[0]][self.drone_pos[1]]=1
        
        self.observation[self.goal_pos[0]][self.goal_pos[1]]=2

        for x,y in self.obstacles_pos:
            self.observation[x][y]=3

        for x,y in self.charging_station_pos:
            self.observation[x][y]=4

        self.observation[self.restaurant_pos[0]][self.restaurant_pos[1]]=5

        self.current_steps=0
        self.max_steps=100

        return self.observation ,{}

        
    #Agent makes a move----> what happens next .... step() helps with that
    def step(self, action):
        self.current_steps +=1
        # Reward for every step
        reward = -1
        old_pos=self.drone_pos.copy()

        # Initially, episode is not terminated
        terminated = False
        truncated = False

        # Action for agent to go up
        if action == 0:
            if self.drone_pos[0] > 0:
                x, y = self.drone_pos
                self.drone_pos[0] = self.drone_pos[0] - 1
                self.observation[x][y] = 0
                self.observation[self.drone_pos[0]][self.drone_pos[1]] = 1
            else:
                pass

        # Action for agent to go down
        elif action == 1:
            if self.drone_pos[0] < self.size - 1:
                x, y = self.drone_pos
                self.drone_pos[0] = self.drone_pos[0] + 1
                self.observation[x][y] = 0
                self.observation[self.drone_pos[0]][self.drone_pos[1]] = 1
            else:
                pass

        # Action for agent to go left
        elif action == 2:
            if self.drone_pos[1] > 0:
                x, y = self.drone_pos
                self.drone_pos[1] = self.drone_pos[1] - 1
                self.observation[x][y] = 0
                self.observation[self.drone_pos[0]][self.drone_pos[1]] = 1
            else:
                pass

        # Action for agent to go right
        elif action == 3:
            if self.drone_pos[1] < self.size - 1:
                x, y = self.drone_pos
                self.drone_pos[1] = self.drone_pos[1] + 1
                self.observation[x][y] = 0
                self.observation[self.drone_pos[0]][self.drone_pos[1]] = 1
            else:
                pass

        if old_pos == self.drone_pos:
            reward = -5
               
        # Check if drone has crashed into an obstacle
        if self.drone_pos in self.obstacles_pos:
            reward = -100
            terminated = True

        # Check if drone has reached the goal
        elif self.drone_pos == self.goal_pos:
            reward = 100
            terminated = True


        if self.current_steps >= self.max_steps:
            truncated = True

        
        return self.observation, reward, terminated, truncated, {}  


        
    
    def render(self):
        pass
        
    def close(self):
        pass

