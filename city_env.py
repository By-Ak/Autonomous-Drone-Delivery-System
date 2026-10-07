import time
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

WORLD_HALF = 30.0                  
MAX_ALT = 30.0                     
ROAD_CENTERS = (-10.0, 10.0)       
ROAD_W = 6.0
STATION_RADIUS = 3.0              
STATION_MAX_SPEED = 3.5           
SAFETY_STOP_TIME = 0.2            
DELIVERY_RADIUS = 1.5

VMAX = 6.0                        
VEL_ALPHA = 0.25                   
SUBSTEPS = 12                     
CONTROL_HZ = 20
LIDAR_RANGE = 12.0
DRONE_R = 0.3


@dataclass
class RewardConfig:
    # --- per-step terms
    time_penalty: float = -0.05      
    progress_coef: float = 0.5      
    gamma: float = 0.995            
    shaping_gamma: float = 1.0      
                                     
    smooth_coef: float = 0.08        
    prox_coef: float = 0.3          
    safe_dist: float = 2.0
    safe_down: float = 1.0
    stall_window: int = 250         
    stall_progress: float = 0.5     
    stall_penalty: float = -100.0
    delivery: float = 50.0
    all_done_bonus: float = 30.0
    charge_per_pct: float = 0.5      
    crash: float = -100.0
    battery_dead: float = -100.0
    out_of_bounds: float = -100.0
    drain_base: float = 0.05
    drain_speed: float = 0.15
    charge_rate: float = 8.0
    reserve_base: float = 15.0       
    reserve_per_m: float = 0.8       
    limit_coef: float = 0.3          # penalty for needing the safety limiter (teaches avoidance)
    low_start_prob: float = 0.35     # fraction of episodes that START with a low battery
    resume_battery: float = 85.0     # above this -> back to deliveries


class CityDroneEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": CONTROL_HZ}

    def __init__(self, render_mode=None, layout_seed=None, layout_pool=(0, 1000),
                 regen_every=1, n_deliveries=3, max_steps=1000, reward_cfg=None,
                 difficulty=1.0, difficulty_mix=False, safety_filter=True):

        super().__init__()
        self.render_mode = render_mode
        self.layout_seed = layout_seed
        self.layout_pool = layout_pool
        self.regen_every = max(1, int(regen_every))
        self._built_seed = None
        self._episodes = 0
        self.current_layout = None
        self.n_deliveries = n_deliveries          # max deliveries per episode
        self.level = float(difficulty)            # difficulty CAP: 0 = easy start, 1 = full task
        self.difficulty = self.level              # difficulty of the current episode
        self.safety_filter = safety_filter
        self.difficulty_mix = difficulty_mix      # train: 30% of episodes are easier than the cap
        self.max_steps = max_steps
        self.cfg = reward_cfg or RewardConfig()

        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, (47,), np.float32)

        mode = p.GUI if render_mode == "human" else p.DIRECT
        self.bc = bullet_client.BulletClient(connection_mode=mode)
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        if render_mode == "human":
            self.bc.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        a16 = np.arange(16) * (np.pi / 8)               # 22.5 deg spacing: thin tree trunks can't hide
        horiz = np.stack([np.cos(a16), np.sin(a16), np.zeros(16)], axis=1)
        a8 = np.arange(8) * (np.pi / 4)
        c45 = np.cos(np.pi / 4)
        elev = np.stack([np.cos(a8) * c45, np.sin(a8) * c45, np.full(8, c45)], axis=1)
        # order: 0-15 horizontal, 16-23 pitched up 45 deg, 24 straight up, 25 straight down
        self.lidar_dirs = np.vstack([horiz, elev, [[0, 0, 1]], [[0, 0, -1]]]).astype(np.float32)

        self.stations = np.zeros((4, 3), np.float32)
        self.pos = np.zeros(3, np.float32)
        self.vel = np.zeros(3, np.float32)
        self.target = np.zeros(3, np.float32)
        self.lidar = np.full(26, LIDAR_RANGE, np.float32)

    def set_difficulty(self, d):
        self.level = float(np.clip(d, 0.0, 1.0))
        self.difficulty = self.level

    def _box(self, half, pos, color, collide=True):
        col = self.bc.createCollisionShape(p.GEOM_BOX, halfExtents=half) if collide else -1
        vis = self.bc.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        return self.bc.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                       baseVisualShapeIndex=vis, basePosition=pos)

    def _tree(self, x, y):
        bc = self.bc
        # trunk
        col = bc.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=3.0)
        vis = bc.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=3.0,
                                   rgbaColor=[0.4, 0.26, 0.13, 1])
        bc.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                           baseVisualShapeIndex=vis, basePosition=[x, y, 1.5])
        # canopy
        col = bc.createCollisionShape(p.GEOM_SPHERE, radius=1.5)
        vis = bc.createVisualShape(p.GEOM_SPHERE, radius=1.5, rgbaColor=[0.1, 0.5, 0.15, 1])
        bc.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                           baseVisualShapeIndex=vis, basePosition=[x, y, 4.2])

    def _build_world(self, layout_seed):
        bc = self.bc
        bc.resetSimulation()                 # wipe the previous city
        self._dbg_line, self._dbg_text = -1, -1
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        bc.setGravity(0, 0, 0)               # velocity-controlled drone, no gravity
        bc.setTimeStep(1.0 / 240.0)
        rng = np.random.default_rng(layout_seed)
        self._built_seed = layout_seed
        park_p = rng.uniform(0.05, 0.30)
        max_h = rng.uniform(12.0, 22.0)
        empty_p = 0.6 * (1.0 - self.difficulty)       # easy levels: many empty lots

        plane = bc.loadURDF("plane.urdf")
        bc.changeVisualShape(plane, -1, rgbaColor=[0.35, 0.55, 0.3, 1], textureUniqueId=-1)

        stations = []
        for qx in (-1, 1):
            for qy in (-1, 1):
                if rng.random() < 0.5:                  # on the vertical road x = +-10
                    stations.append([qx * 10.0, qy * rng.uniform(2.0, 26.0), 1.0])
                else:                                   # on the horizontal road y = +-10
                    stations.append([qx * rng.uniform(2.0, 26.0), qy * 10.0, 1.0])
        self.stations = np.array(stations, dtype=np.float32)

        def near_station(x, y):
            return bool(np.any(np.hypot(self.stations[:, 0] - x, self.stations[:, 1] - y) < 4.5))

        for c in ROAD_CENTERS:
            self._box([ROAD_W / 2, WORLD_HALF, 0.01], [c, 0, 0.01], [0.2, 0.2, 0.22, 1], False)
            self._box([WORLD_HALF, ROAD_W / 2, 0.01], [0, c, 0.012], [0.2, 0.2, 0.22, 1], False)

        for bx in (-20.0, 0.0, 20.0):
            for by in (-20.0, 0.0, 20.0):
                for dx in (-3.5, 3.5):
                    for dy in (-3.5, 3.5):
                        cx, cy = bx + dx, by + dy
                        r = rng.random()
                        if r < empty_p:                 # empty lot (easy levels only)
                            continue
                        if r < empty_p + park_p:        # park
                            self._tree(cx, cy)
                            continue
                        h = rng.uniform(5.0, 6.0 + (max_h - 6.0) * self.difficulty)
                        half = [rng.uniform(2.2, 3.0), rng.uniform(2.2, 3.0), h / 2]
                        shade = rng.uniform(0.45, 0.75)
                        self._box(half, [cx, cy, h / 2], [shade, shade, shade + 0.1, 1])

        # trees along roads (away from intersections)
        for c in ROAD_CENTERS:
            for t in range(-24, 25, 8):
                if any(abs(t - r) < 6 for r in ROAD_CENTERS):
                    continue
                for tx, ty in ((c + rng.choice([-1, 1]) * 2.8, t),
                               (t, c + rng.choice([-1, 1]) * 2.8)):
                    if not near_station(tx, ty):
                        self._tree(tx, ty)

        # charging station visuals (charging itself is handled in step())
        for s in self.stations:
            vis = bc.createVisualShape(p.GEOM_CYLINDER, radius=STATION_RADIUS, length=0.1,
                                       rgbaColor=[0.1, 0.9, 0.3, 1])
            bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                               basePosition=[s[0], s[1], 0.06])
            vis = bc.createVisualShape(p.GEOM_CYLINDER, radius=0.15, length=3.0,
                                       rgbaColor=[0.1, 0.9, 0.3, 0.6])
            bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                               basePosition=[s[0], s[1], 1.5])

        # delivery target marker (visual only, moved each delivery)
        vis = bc.createVisualShape(p.GEOM_SPHERE, radius=0.6, rgbaColor=[1.0, 0.6, 0.0, 0.8])
        self.marker = bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                         basePosition=[0, 0, 2])

        # the drone
        col = bc.createCollisionShape(p.GEOM_SPHERE, radius=DRONE_R)
        vis = bc.createVisualShape(p.GEOM_SPHERE, radius=DRONE_R, rgbaColor=[0.9, 0.1, 0.1, 1])
        self.drone = bc.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis, basePosition=[0, 0, 2])

    def _sample_road_point(self, z):
        c = self.np_random.choice(ROAD_CENTERS)
        along = self.np_random.uniform(-26, 26)
        off = self.np_random.uniform(-0.6, 0.6)
        if self.np_random.random() < 0.5:
            return np.array([c + off, along, z], np.float32)
        return np.array([along, c + off, z], np.float32)

    def _new_target(self):
        max_d = 12.0 + 68.0 * self.difficulty
        best, best_gap = None, 1e9
        for _ in range(100):
            t = self._sample_road_point(self.np_random.uniform(1.5, 3.0))
            d = float(np.linalg.norm(t - self.pos))
            if 6.0 <= d <= max_d:
                best = t
                break
            gap = (6.0 - d) if d < 6.0 else (d - max_d)
            if gap < best_gap:
                best, best_gap = t, gap
        t = best
        self.target = t
        self.bc.resetBasePositionAndOrientation(self.marker, t.tolist(), [0, 0, 0, 1])

    def _nearest_station(self):
        return int(np.argmin(np.linalg.norm(self.stations - self.pos, axis=1)))

    def _goal_pos(self):
        return self.stations[self.station_idx] if self.seeking else self.target

    def _reserve(self):
        d = float(np.linalg.norm(self.stations - self.pos, axis=1).min())
        return self.cfg.reserve_base + self.cfg.reserve_per_m * d

    def _update_goal(self):
        cfg = self.cfg
        if not self.seeking and self.battery < self._reserve():
            self.seeking = True
            self.station_idx = self._nearest_station()
        elif self.seeking and self.battery >= cfg.resume_battery:
            self.seeking = False
            self.charges += 1
        key = ("station", self.station_idx) if self.seeking else ("delivery", self.deliveries)
        if key != self.goal_key:
            self.goal_key = key
            self.prev_dist = float(np.linalg.norm(self._goal_pos() - self.pos))
            self.best_dist = self.prev_dist
            self.stall_steps = 0

    def _lidar(self):
        off = DRONE_R + 0.05
        starts = [(self.pos + d * off).tolist() for d in self.lidar_dirs]
        ends = [(self.pos + d * LIDAR_RANGE).tolist() for d in self.lidar_dirs]
        res = self.bc.rayTestBatch(starts, ends)
        return np.array([off + r[2] * (LIDAR_RANGE - off) for r in res], np.float32)

    def _safety_limit(self, vel):
        """Collision-avoidance layer: for every lidar ray, cap the velocity component pointing
        along it so the drone can always stop before the obstacle (stop distance margin 0.8 m
        from the centre). Sliding along walls is still allowed."""
        vel = vel.copy()
        allowed = np.maximum(0.0, (self.lidar - 0.8) / SAFETY_STOP_TIME)
        for _ in range(2):
            for u, lim in zip(self.lidar_dirs, allowed):
                vp = float(vel @ u)
                if vp > lim:
                    vel -= (vp - lim) * u
        return vel.astype(np.float32)

    def _goal_feat(self, goal):
        """Unit direction (full-strength signal at any range) + distance/40 m."""
        v = goal - self.pos
        d = float(np.linalg.norm(v))
        return np.concatenate([v / max(d, 1e-6), [min(d / 40.0, 1.0)]])

    def _obs(self):
        pos = self.pos
        stn = self.stations[self.station_idx] if self.seeking else self.stations[self._nearest_station()]
        obs = np.concatenate([
            self._goal_feat(self.target),
            self._goal_feat(stn),
            [self.battery / 100.0],
            [(self.battery - self._reserve()) / 50.0],   # >0: safe, <0: should be charging
            self.vel / VMAX,
            [pos[0] / WORLD_HALF, pos[1] / WORLD_HALF, pos[2] / MAX_ALT],
            [float(self.seeking)],
            self.lidar / LIDAR_RANGE,
            [min(1.0, self.stall_steps / self.cfg.stall_window)],
            self.prev_action,
        ])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.difficulty_mix and self.np_random.random() < 0.3:
            self.difficulty = float(self.np_random.uniform(0.0, self.level))   # revisit easier levels
        else:
            self.difficulty = self.level
        if self.layout_seed is not None:                       # fixed city
            if self._built_seed != self.layout_seed:
                self._build_world(self.layout_seed)
        elif self._built_seed is None or self._episodes % self.regen_every == 0:
            lo, hi = self.layout_pool                          # new random city
            self._build_world(int(self.np_random.integers(lo, hi)))
        self._episodes += 1
        self.current_layout = self._built_seed

        self.pos = self._sample_road_point(2.0)
        self.vel = np.zeros(3, np.float32)
        self.bc.resetBasePositionAndOrientation(self.drone, self.pos.tolist(), [0, 0, 0, 1])
        self.bc.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])

        lo = 80.0 - 45.0 * self.difficulty                       # 80..100% easy, 35..100% full
        self.battery = float(self.np_random.uniform(lo, 100.0))
        if self.difficulty >= 0.2 and self.np_random.random() < self.cfg.low_start_prob:
            self.battery = float(self.np_random.uniform(25.0, 40.0))   # practise charging often
        self.charges = 0
        self.n_cur = min(self.n_deliveries,
                         1 + int(self.difficulty * 3 >= 1.0) + int(self.difficulty * 3 >= 2.0))
        self.steps = 0
        self.deliveries = 0
        self.prev_action = np.zeros(3, np.float32)
        self.seeking = False
        self.station_idx = 0
        self.goal_key = None
        self.stall_steps = 0
        self.best_dist = 1e9
        self._new_target()
        self.lidar = self._lidar()
        self._update_goal()
        return self._obs(), {}

    def step(self, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)

        # ---- move the drone (smoothed velocity command), check collisions
        self.vel += VEL_ALPHA * (action * VMAX - self.vel)
        limited = 0.0
        if self.safety_filter:
            before = self.vel.copy()
            self.vel = self._safety_limit(self.vel)
            limited = float(np.linalg.norm(before - self.vel)) / VMAX
        crashed = False
        for _ in range(SUBSTEPS):
            self.bc.resetBaseVelocity(self.drone, linearVelocity=self.vel.tolist(),
                                      angularVelocity=[0, 0, 0])
            self.bc.stepSimulation()
            if self.bc.getContactPoints(bodyA=self.drone):
                crashed = True
                break
        self.pos = np.array(self.bc.getBasePositionAndOrientation(self.drone)[0], np.float32)
        self.steps += 1
        speed = float(np.linalg.norm(self.vel))
        self.lidar = self._lidar()

        # ---- battery drain
        self.battery = max(0.0, self.battery - (cfg.drain_base + cfg.drain_speed * (speed / VMAX) ** 2))

        # ---- dense reward terms
        reward = cfg.time_penalty
        dist = float(np.linalg.norm(self._goal_pos() - self.pos))
        reward += cfg.progress_coef * (self.prev_dist - cfg.shaping_gamma * dist)  # progress
        self.prev_dist = dist
        if dist < self.best_dist - cfg.stall_progress:     # real progress toward the goal
            self.best_dist = dist
            self.stall_steps = 0
        else:
            self.stall_steps += 1
        reward -= cfg.smooth_coef * float(np.linalg.norm(action - self.prev_action))
        self.prev_action = action
        front = max(0.0, 1.0 - float(self.lidar[:25].min()) / cfg.safe_dist)
        down = max(0.0, 1.0 - float(self.lidar[25]) / cfg.safe_down)
        reward -= cfg.prox_coef * (front ** 2 + down ** 2)
        reward -= cfg.limit_coef * limited 

        terminated, truncated, event = False, False, None
        oob = (abs(self.pos[0]) > WORLD_HALF or abs(self.pos[1]) > WORLD_HALF
               or self.pos[2] > MAX_ALT)

        if crashed:
            reward += cfg.crash; terminated = True; event = "crash"
        elif self.battery <= 0.0:
            reward += cfg.battery_dead; terminated = True; event = "battery_dead"
        elif oob:
            reward += cfg.out_of_bounds; terminated = True; event = "out_of_bounds"
        elif self.stall_steps >= cfg.stall_window:
            reward += cfg.stall_penalty; terminated = True; event = "stalled"
        else:
            # charging: only inside a station zone and moving slowly
            near = np.linalg.norm(self.stations - self.pos, axis=1).min() < STATION_RADIUS
            if near and speed < STATION_MAX_SPEED and self.battery < 100.0:
                gain = min(cfg.charge_rate, 100.0 - self.battery)
                if self.seeking:                      # only needed charging pays (anti-farming)
                    reward += cfg.charge_per_pct * gain
                self.battery += gain

            # delivery
            if np.linalg.norm(self.target - self.pos) < DELIVERY_RADIUS:
                reward += cfg.delivery
                self.deliveries += 1
                if self.deliveries >= self.n_cur:
                    reward += cfg.all_done_bonus
                    terminated = True; event = "success"
                else:
                    self._new_target()

            if not terminated and self.steps >= self.max_steps:
                truncated = True; event = "timeout"

            if not terminated:
                self._update_goal()

        info = {"deliveries": self.deliveries, "battery": self.battery,
                "n_deliveries": self.n_cur, "difficulty": self.difficulty,
                "charges": self.charges}
        if event:
            info["event"] = event

        if self.render_mode == "human":
            self.bc.resetDebugVisualizerCamera(28, 45, -40, self.pos.tolist())
            # overlay: line to the CURRENT goal (green = charger, orange = delivery) + status
            color = [0.1, 0.9, 0.3] if self.seeking else [1.0, 0.6, 0.0]
            self._dbg_line = self.bc.addUserDebugLine(
                self.pos.tolist(), self._goal_pos().tolist(), lineColorRGB=color, lineWidth=2.0,
                replaceItemUniqueId=self._dbg_line)
            label = (f"{'GO CHARGE' if self.seeking else 'DELIVER'} | battery {self.battery:.0f}% "
                     f"| deliveries {self.deliveries}/{self.n_cur}")
            self._dbg_text = self.bc.addUserDebugText(
                label, (self.pos + np.array([0, 0, 1.5], np.float32)).tolist(),
                textColorRGB=[1, 1, 1], textSize=1.5, replaceItemUniqueId=self._dbg_text)
            time.sleep(1.0 / CONTROL_HZ)

        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.bc.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    check_env(CityDroneEnv(), warn=True)
    print("check_env passed")

    env = CityDroneEnv(render_mode="human")
    obs, _ = env.reset(seed=0)
    for _ in range(2000):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            print("episode ended:", info)
            obs, _ = env.reset()
    env.close()
