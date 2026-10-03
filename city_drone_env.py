import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import random

class CityDroneEnv(gym.Env):
    metadata = {'render_modes': ['human', 'direct']}

    def __init__(self, render_mode="direct"):
        super().__init__()
        self.render_mode = render_mode
        
        if self.render_mode == "human":
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Action Space: 0: Hover, 1: Forward, 2: Backward, 3: Left, 4: Right, 5: Up, 6: Down
        self.action_space = spaces.Discrete(7)
        
        # Observation: [x, y, z, battery, dist_to_target, dist_to_c1, dist_to_c2, dist_to_c3]
        # Expanded bounds to handle a 100x100 city map
        low = np.array([-60, -60, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([60, 60, 100, 250, 300, 300, 300, 300], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Far-end spawn and target points to force long-distance routing
        self.start_pos = np.array([-45.0, -45.0, 1.0])
        self.target_pos = np.array([45.0, 45.0, 1.0])
        
        # Spread 3 charging stations across different city districts
        self.chargers = [
            np.array([-20.0, 20.0, 0.1]),
            np.array([20.0, -20.0, 0.1]),
            np.array([0.0, 0.0, 0.1])
        ]
        self.building_ids = []

    def _create_box(self, position, half_extents, color):
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=position)

    def _create_cylinder(self, position, radius, height, color):
        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=position)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Pull the camera way back to see the entire metropolis
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=85,
                cameraYaw=45,
                cameraPitch=-60,
                cameraTargetPosition=[0, 0, 5]
            )
        
        # Load a large asphalt-colored ground plane
        self.planeId = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.planeId, -1, rgbaColor=[0.2, 0.2, 0.2, 1])
        
        # Spawn Target (Red Hub) and Chargers (Green Hubs)
        self._create_cylinder(self.target_pos, radius=2.0, height=0.2, color=[0.9, 0.1, 0.1, 1])
        self._create_box([self.target_pos[0], self.target_pos[1], 3.0], [0.2, 0.2, 2.0], [1, 0, 0, 1])
        
        for c_pos in self.chargers:
            self._create_cylinder(c_pos, radius=2.5, height=0.2, color=[0.1, 0.9, 0.2, 1])
            self._create_box([c_pos[0], c_pos[1], 1.0], [0.2, 0.2, 1.0], [0, 1, 0, 1])
            
        # Procedurally generate city blocks from x=-40 to 40 and y=-40 to 40
        self.building_ids = []
        city_colors = [[0.3, 0.4, 0.5, 1], [0.4, 0.4, 0.4, 1], [0.25, 0.25, 0.28, 1], [0.45, 0.5, 0.55, 1]]
        
        # We loop through a grid, stepping by 15 units to create city "blocks"
        for x in range(-35, 45, 15):
            for y in range(-35, 45, 15):
                # Leave empty space around the chargers and target/spawn so they don't get buried in buildings
                dist_to_start = np.linalg.norm([x - self.start_pos[0], y - self.start_pos[1]])
                dist_to_target = np.linalg.norm([x - self.target_pos[0], y - self.target_pos[1]])
                
                if dist_to_start < 10 or dist_to_target < 10:
                    continue
                
                skip_charger = False
                for c in self.chargers:
                    if np.linalg.norm([x - c[0], y - c[1]]) < 8:
                        skip_charger = True
                if skip_charger:
                    continue
                
                # Randomize skyscraper dimensions
                w = random.uniform(2.0, 4.5)  # Width
                d = random.uniform(2.0, 4.5)  # Depth
                h = random.uniform(4.0, 20.0) # Height (Towers can get very tall)
                
                color = random.choice(city_colors)
                
                # Add slight random offset inside the block so it looks natural
                b_x = x + random.uniform(-1.5, 1.5)
                b_y = y + random.uniform(-1.5, 1.5)
                
                b_id = self._create_box([b_x, b_y, h], [w, d, h], color)
                self.building_ids.append(b_id)
            
        # Spawn Drone (Bright Orange Cube)
        col_drone = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.2])
        vis_drone = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.2], rgbaColor=[1.0, 0.6, 0.0, 1])
        self.droneId = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col_drone, baseVisualShapeIndex=vis_drone, basePosition=self.start_pos)
        
        # Increased battery capacity for the massive map
        self.battery = 250.0
        self.step_count = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.droneId)
        dist_to_target = np.linalg.norm(np.array(pos) - self.target_pos)
        dist_c1 = np.linalg.norm(np.array(pos) - self.chargers[0])
        dist_c2 = np.linalg.norm(np.array(pos) - self.chargers[1])
        dist_c3 = np.linalg.norm(np.array(pos) - self.chargers[2])
        return np.array([pos[0], pos[1], pos[2], self.battery, dist_to_target, dist_c1, dist_c2, dist_c3], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        pos, _ = p.getBasePositionAndOrientation(self.droneId)
        
        force = [0, 0, 0]
        thrust = 9.81
        
        # Increased movement speed multiplier to traverse the large map faster
        speed = 12
        if action == 1: force = [speed, 0, thrust]
        elif action == 2: force = [-speed, 0, thrust]
        elif action == 3: force = [0, speed, thrust]
        elif action == 4: force = [0, -speed, thrust]
        elif action == 5: force = [0, 0, thrust + speed]
        elif action == 6: force = [0, 0, max(0, thrust - speed + 2)]
        elif action == 0: force = [0, 0, thrust]
        
        p.applyExternalForce(self.droneId, -1, force, pos, p.WORLD_FRAME)
        p.stepSimulation()
        
        # Battery drain rate
        self.battery -= 0.15
        
        new_pos, _ = p.getBasePositionAndOrientation(self.droneId)
        dist_to_target = np.linalg.norm(np.array(new_pos) - self.target_pos)
        
        # Charging station verification
        for c_pos in self.chargers:
            dist_to_c = np.linalg.norm(np.array(new_pos) - c_pos)
            if dist_to_c < 2.5 and new_pos[2] < 1.5:
                self.battery = min(250.0, self.battery + 25.0)
        
        reward = -0.1
        terminated = False
        
        if dist_to_target < 2.5:
            reward = 1000.0  # Increased reward for crossing the massive map
            terminated = True
        elif self.battery <= 0:
            reward = -200.0
            terminated = True
            
        # Hard bounds logic so it doesn't fly infinitely into the void
        if abs(new_pos[0]) > 55 or abs(new_pos[1]) > 55 or new_pos[2] > 40:
            reward = -200.0
            terminated = True
            
        for b_id in self.building_ids:
            if len(p.getContactPoints(self.droneId, b_id)) > 0:
                reward = -200.0
                terminated = True
                break
                
        # Increased maximum steps before truncation
        truncated = self.step_count > 3000
        
        return self._get_obs(), reward, terminated, truncated, {}