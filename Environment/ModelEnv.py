import gymnasium as gym
import numpy as np

class Cityenv(gym.Env):   # Defining how our environment will look like 
    def __init__(self):
        super().__init__()
        self.size=np.array([2096,2096,1048])            # Initializing map size
        

        # Defining number of actions that will be used
        self.action_space=gym.spaces.Discrete(6)   

        # Defines what the agent can observe: city map, restaurant position, drone position, goal position,
        # charging station position, velocity,nearby obstacle, remaining battery percentage
        
        self.observation_space=gym.spaces.Dict(
            {
                "map":gym.spaces.Box(
                    low=0 , high=1 , shape=(131,131,66) , dtype=np.float32     #[2096, 2096, 1048] / 16 = 131,131,66 ---> size down for representing grid
                ), 

                "drone_pos":gym.spaces.Box(
                    low=np.array([0,0,0]) , high=np.array([2096,2096,200]) , shape=(3,) , dtype=np.float32     
                ),

                "battery":gym.spaces.Box(
                    low=0 , high=100 , shape=(1,) , dtype=np.float32
                ),

                "goal_pos":gym.spaces.Box(
                    low=np.array([0,0,0]) , high=np.array([2096,2096,200]) , shape=(3,) , dtype=np.float32
                ),

                "restaurant_pos":gym.spaces.Box(
                    low=np.array([0,0,0]) , high=np.array([2096,2096,200]) , shape=(3,) , dtype=np.float32
                ),

                "charging_station_pos":gym.spaces.Box(
                    low=np.array([0,0,0]) , high=np.array([2096,2096,200]) , shape=(3,) , dtype=np.float32
                ),

                "nearby_obstacle":gym.spaces.Box(
                    low=0 , high=100 , shape=(6,) , dtype=np.float32  # shape 6 because it will observe obs to its [LEFT, RIGHT , FORWARD, BACKWARD , UP ,DOWN]
                ),    
            })  
        

        
#------------------------------------------------------------------------------------------------------------

    #defining how our self.observation will look like after reset or at start of an episode
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)

        #Defining al the values 
        self.battery= np.array([100], dtype=np.float32)
        self.drone_pos= [0,0,0]
        self.goal_pos= [0,0,0]
        self.restaurant_pos= [0,0,0]
        self.nearby_obstacle= [100,100,100,100,100,100] 
        self.charging_station_pos= [0,0,0]
        self.obstacles_pos= [0,0,0]

        #---------------------------------------------------------------------------

        #Inserting all the values in our observation variable 

        self.observation={
            "map" : np.zeros((131,131,66),dtype=np.float32),
            "drone_pos" : self.drone_pos,
            "restaurant_pos" : self.restaurant_pos,
            "goal_pos" : self.goal_pos,
            "battery" : self.battery,
            "charging_station_pos" : self.charging_station_pos,
            "nearby_obstacle" : self.get_nearby_obstacles()

            }

        #---------------------------------------------------------------------------

        self.current_steps=0
        self.max_steps=3000   

        return self.observation ,{}

#-----------------------------------------------------------------------------------------------

    # Detect all obstacles nearby to our drone 

    def get_nearby_obstacles(self):

        drone = np.array(self.drone_pos)

        # Default distance = 100
        left = 100
        right = 100
        forward = 100
        backward = 100
        up = 100
        down = 100

        for obstacle in self.obstacles_pos:

            obstacle = np.array(obstacle)

            dx = obstacle[0] - drone[0]
            dy = obstacle[1] - drone[1]
            dz = obstacle[2] - drone[2]

            distance = np.linalg.norm(obstacle - drone)

            # LEFT
            if dx < 0:
                left = min(left, distance)

            # RIGHT
            elif dx > 0:
                right = min(right, distance)

            # FORWARD
            if dy > 0:
                forward = min(forward, distance)

            # BACKWARD
            elif dy < 0:
                backward = min(backward, distance)

            # UP
            if dz > 0:
                up = min(up, distance)

            # DOWN
            elif dz < 0:
                down = min(down, distance)

        return np.array(
            [left, right, forward, backward, up, down],
            dtype=np.float32
    )

#-----------------------------------------------------------------------------------------------
        
    #Agent makes a move----> what happens next .... step() helps with that
    def step(self, action):

        #Increment steps to ensure if our agent get stuck somewhere --> truncate episode 
        self.current_steps +=1

        # Reward for every step
        reward = -1

        old_pos=self.observation["drone_pos"].copy()

        # Initially, episode is not terminated
        terminated = False
        truncated = False

        # Action for agent to go up
        if action == 0:
            if self.observation["drone_pos"][2] < 199 :

                self.observation["drone_pos"][2] = self.observation["drone_pos"][2] + 1
                
            else:
                pass

        # Action for agent to go down
        elif action == 1:
            if self.observation["drone_pos"][2] > 0:

                self.observation["drone_pos"][2] = self.observation["drone_pos"][2] - 1
                
            else:
                pass

        # Action for agent to go left
        elif action == 2:
            if self.observation["drone_pos"][0] > 0:

                self.observation["drone_pos"][0] -= 1

            else:
                pass

        # Action for agent to go right
        elif action == 3:
            if self.observation["drone_pos"][0] < 2095:
                self.observation["drone_pos"][0] += 1
            else:
                pass


        #Action for agent to go forward
        elif action == 4:
            if self.observation["drone_pos"][1] < 2095:
                self.observation["drone_pos"][1] += 1

        #Action for agent to go backward
        elif action == 5:
            if self.observation["drone_pos"][1] > 0:
                self.observation["drone_pos"][1] -= 1


        #Get new observation for nearby obstacle
        self.observation["nearby_obstacle"] = self.get_nearby_obstacles()

        #If drone at same position after an action penalize it
        if old_pos == self.observation["drone_pos"]:
            reward = -5
               
        # Check if drone has crashed into an obstacle
        if self.observation["drone_pos"] in self.obstacles_pos:
            reward = -200
            terminated = True

        # Check if drone has reached the goal
        elif self.observation["drone_pos"] == self.observation["goal_pos"]:
            reward = 100

        # Check if drone has reached the restaurant
        elif self.observation["drone_pos"] == self.observation["restaurant_pos"]:
            reward = 200
            terminated = True

        if self.current_steps >= self.max_steps:
            truncated = True

        
        return self.observation, reward, terminated, truncated, {}  


#------------------------------------------------------------------------------------------
        
    
    def render(self):
        pass

#-----------------------------------------------------------------------------------------
     
    def close(self):
        pass

