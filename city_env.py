"""
city_env.py  -  PyBullet city delivery-drone environment (Gymnasium API)

World: 60 x 60 m city, 3x3 blocks of buildings/parks, roads between blocks,
trees along roads, 4 charging stations placed randomly ON the roads (one per
quadrant) in every new city.
Drone: point-mass sphere controlled by velocity commands (real PyBullet
collisions with buildings, trees and ground).

Run this file directly to sanity-check the env and watch a random drone.
"""
import heapq
import time
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

# ----------------------------------------------------------------- constants
N_BLOCKS = 4                       # city = N_BLOCKS x N_BLOCKS blocks (4 -> 80 x 80 m, 3 -> 60 x 60 m)
PITCH = 20.0                       # spacing between block centres
WORLD_HALF = N_BLOCKS * PITCH / 2  # world is [-WORLD_HALF, WORLD_HALF] in x and y
BLOCK_CENTERS = tuple((k - (N_BLOCKS - 1) / 2) * PITCH for k in range(N_BLOCKS))
ROAD_CENTERS = tuple(b + PITCH / 2 for b in BLOCK_CENTERS[:-1])   # roads between blocks
STATIONS_PER_QUADRANT = 2
N_STATIONS = 4 * STATIONS_PER_QUADRANT
MAX_ALT = 30.0                     # ceiling (tallest building is 22 m)
ROAD_W = 6.0

# ---- road-network route planner ------------------------------------------------
# Every start/goal lies on a road, and roads are free corridors between the buildings, so
# the drone gets a collision-free ROUTE (list of waypoints along roads) as guidance.
ROAD_LINES = [("V", i) for i in range(len(ROAD_CENTERS))] + [("H", j) for j in range(len(ROAD_CENTERS))]
CRUISE_Z = 2.5                     # altitude of route corner points
WP_RADIUS = 2.5                    # a waypoint counts as reached within this (horizontal) distance
OFFROUTE_DIST = 8.0                # farther than this from the current route leg -> replan
BLOCK_COST = 300.0                 # extra cost (m) of a road segment/direction that was "blocked"


def _snap_to_road(xy):
    """Nearest road line to a point -> (line_id, coordinate along that line, port_xy)."""
    best = None
    for line in ROAD_LINES:
        kind, k = line
        r = ROAD_CENTERS[k]
        if kind == "V":                                   # vertical road x = r
            perp, along = abs(xy[0] - r), float(np.clip(xy[1], -WORLD_HALF, WORLD_HALF))
            port = (r, along)
        else:                                             # horizontal road y = r
            perp, along = abs(xy[1] - r), float(np.clip(xy[0], -WORLD_HALF, WORLD_HALF))
            port = (along, r)
        if best is None or perp < best[0]:
            best = (perp, line, along, port)
    return best[1], best[2], best[3]


def plan_road_route(start_xy, goal_xy, blocked=()):
    """Shortest route along the road grid (Dijkstra). Returns [(x, y, line_of_arriving_leg), ...]
    from the start's road port to the goal's road port, colinear points removed.
    `blocked` holds (line, segment, direction) entries that cost BLOCK_COST extra."""
    s_line, s_c, s_port = _snap_to_road(start_xy)
    g_line, g_c, g_port = _snap_to_road(goal_xy)
    n = len(ROAD_CENTERS)
    stops = {line: [] for line in ROAD_LINES}
    for i in range(n):                                    # V road i crosses H road j at (R[i], R[j])
        for j in range(n):
            node = ("I", i, j)
            stops[("V", i)].append((ROAD_CENTERS[j], node))
            stops[("H", j)].append((ROAD_CENTERS[i], node))
    stops[s_line].append((s_c, "S"))
    stops[g_line].append((g_c, "G"))

    adj = {}
    for line, lst in stops.items():
        lst.sort(key=lambda t: t[0])
        for (c1, n1), (c2, n2) in zip(lst[:-1], lst[1:]):
            length = c2 - c1
            seg = int(np.searchsorted(ROAD_CENTERS, 0.5 * (c1 + c2)))
            for a, b, d in ((n1, n2, 1), (n2, n1, -1)):
                cost = length + (BLOCK_COST if (line, seg, d) in blocked else 0.0)
                adj.setdefault(a, []).append((b, cost, line))

    def xy_of(node):
        if node == "S":
            return s_port
        if node == "G":
            return g_port
        return (ROAD_CENTERS[node[1]], ROAD_CENTERS[node[2]])

    dist, prev, heap, cnt = {"S": 0.0}, {}, [(0.0, 0, "S")], 1
    while heap:
        d, _, u = heapq.heappop(heap)
        if u == "G":
            break
        if d > dist.get(u, 1e18):
            continue
        for v, w, line in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                prev[v] = (u, line)
                heapq.heappush(heap, (nd, cnt, v))
                cnt += 1
    nodes, lines, u = [], [], "G"
    while u != "S":
        pu, line = prev[u]
        nodes.append(u); lines.append(line)
        u = pu
    nodes.append("S"); lines.append(s_line)
    nodes.reverse(); lines.reverse()

    out = []
    for k, (node, line) in enumerate(zip(nodes, lines)):
        if 0 < k < len(nodes) - 1 and lines[k] == lines[k + 1]:
            continue                                      # straight through: not a corner
        x, y = xy_of(node)
        out.append((float(x), float(y), line))
    return out
STATION_RADIUS = 3.0               # must be within this (3D) to charge
STATION_MAX_SPEED = 3.5            # ...and slower than this
SAFETY_STOP_TIME = 0.2             # s; used by the collision-avoidance speed limiter
DELIVERY_RADIUS = 1.5
RESTAURANT_RADIUS = 2.5            # within this (3D) of the pad = at the restaurant

VMAX = 6.0                         # m/s
VEL_ALPHA = 0.25                   # velocity smoothing (fake inertia)
SUBSTEPS = 12                      # 240 Hz physics / 12 = 20 Hz control
CONTROL_HZ = 20
LIDAR_RANGE = 12.0
DRONE_R = 0.3


@dataclass
class RewardConfig:
    # --- per-step terms
    time_penalty: float = -0.05      # every step costs a little
    progress_coef: float = 0.3       # reward per metre of ROUTE progress
    gamma: float = 0.995             # single source of truth: train.py reads this for PPO
    shaping_gamma: float = 1.0       # 1.0 = pure progress reward: total shaping telescopes to
                                     # coef*(start_dist - end_dist), so hovering earns nothing
    smooth_coef: float = 0.08        # penalises jittery actions (change between steps)
    prox_coef: float = 0.3           # penalises flying too close to obstacles
    safe_dist: float = 2.0
    safe_down: float = 1.0
    # --- anti-stall: if the drone makes no progress toward its goal for this long, fail it
    stall_window: int = 250          # steps (12.5 s)
    stall_progress: float = 0.5      # metres of NEW best distance that count as progress
    stall_penalty: float = -100.0
    # --- hovering in one place: growing penalty, escape bonus, then re-route onto a different road
    dwell_radius: float = 2.0        # "staying in place" = within this distance of the anchor
    dwell_grace: int = 30            # steps before the dwell penalty starts (1.5 s)
    dwell_ramp: int = 50             # steps for the penalty to grow to full strength
    dwell_coef: float = 0.3          # max extra penalty per step
    escape_bonus: float = 0.8        # one-off bonus for breaking out after being stuck
    reroute_after: int = 80          # dwelling this long -> plan a different route (4 s)
    reroute_every: int = 60          # ...and again every this many steps while still stuck    # --- events
    delivery: float = 50.0
    all_done_bonus: float = 30.0     # all parcels delivered AND drone back at the restaurant
    return_bonus: float = 10.0       # back at the restaurant after a delivery
    trip_per_m: float = 0.6          # battery % per metre assumed when planning a delivery trip
    trip_margin: float = 8.0         # extra % kept in reserve for a trip
    charge_per_pct: float = 0.5      # per % gained, ONLY while in seek-charger mode (anti-farming)
    crash: float = -100.0
    battery_dead: float = -100.0
    out_of_bounds: float = -100.0
    # --- battery model (percent per control step)
    drain_base: float = 0.04
    drain_speed: float = 0.10
    charge_rate: float = 1.0         # % per step: charging is SLOW (~55 steps for 30% -> 85%)
    reserve_base: float = 15.0       # seek-charger mode starts when battery < reserve_base
    reserve_per_m: float = 0.6       #   + reserve_per_m * (metres to nearest station)
    limit_coef: float = 0.3          # penalty for needing the safety limiter (teaches avoidance)
    low_start_prob: float = 0.5      # fraction of episodes that START with a low battery
    charge_start_bonus: float = 10.0 # once per charge cycle: reached a charger and started charging
    urgency_coef: float = 0.2        # per-step cost while seeking a charger; grows as battery drops
    dock_gain: float = 0.8           # docking assist: while seeking, max speed = dock_gain * distance
    dock_min_speed: float = 1.5      #   to the charger (but never below this) so it can actually charge
    resume_battery: float = 85.0     # above this -> back to deliveries


class CityDroneEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": CONTROL_HZ}

    def __init__(self, render_mode=None, layout_seed=None, layout_pool=(0, 1000),
                 regen_every=1, n_deliveries=3, max_steps=2000, reward_cfg=None,
                 difficulty=1.0, difficulty_mix=False, safety_filter=True):
        """
        layout_seed : int -> one fixed city forever.
                      None -> a NEW random city is generated during reset(), drawn
                              from layout_pool = (low, high) seeds.
        regen_every : with layout_seed=None, rebuild the city every N episodes.
        Keep train pools and evaluation pools disjoint (e.g. 0-999 vs 10000-10099).
        """
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
        # target(4) + station(4) + restaurant(4) + route(9) + battery1 + battery-margin1 + vel3 + pos3
        # + seeking1 + carrying1 + lidar26 + stall timer1 + dwell timer1 + previous action3 = 62
        self.observation_space = spaces.Box(-1.0, 1.0, (62,), np.float32)

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

        self.stations = np.zeros((N_STATIONS, 3), np.float32)
        self.restaurant = np.zeros(3, np.float32)
        self.carrying = False
        self.awaiting_return = False
        self.pos = np.zeros(3, np.float32)
        self.vel = np.zeros(3, np.float32)
        self.target = np.zeros(3, np.float32)
        self.lidar = np.full(26, LIDAR_RANGE, np.float32)

    def set_difficulty(self, d):
        """Curriculum hook (called from train.py through env_method). Takes effect
        at the next reset(): d=0 -> sparse low city, close targets, 1 delivery, full
        battery. d=1 -> the full task."""
        self.level = float(np.clip(d, 0.0, 1.0))
        self.difficulty = self.level

    # ------------------------------------------------------------ world build
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
        self._dbg_line, self._dbg_text, self._dbg_speed = -1, -1, -1
        self._dbg_route = []
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        bc.setGravity(0, 0, 0)               # velocity-controlled drone, no gravity
        bc.setTimeStep(1.0 / 240.0)
        rng = np.random.default_rng(layout_seed)
        self._built_seed = layout_seed
        # per-city style: park density and how tall the skyline is
        park_p = rng.uniform(0.05, 0.30)
        max_h = rng.uniform(12.0, 22.0)
        empty_p = 0.6 * (1.0 - self.difficulty)       # easy levels: many empty lots

        plane = bc.loadURDF("plane.urdf")
        bc.changeVisualShape(plane, -1, rgbaColor=[0.35, 0.55, 0.3, 1], textureUniqueId=-1)

        # charging stations: STATIONS_PER_QUADRANT per quadrant, random spots on road centre lines
        stations = []
        for qx in (-1, 1):
            for qy in (-1, 1):
                for _ in range(STATIONS_PER_QUADRANT):
                    for _try in range(30):
                        along = rng.uniform(2.0, WORLD_HALF - 4.0)
                        if rng.random() < 0.5:                       # on a vertical road
                            road = rng.choice([r for r in ROAD_CENTERS if r == 0 or np.sign(r) == qx])
                            s = [float(road), qy * along, 1.0]
                        else:                                        # on a horizontal road
                            road = rng.choice([r for r in ROAD_CENTERS if r == 0 or np.sign(r) == qy])
                            s = [qx * along, float(road), 1.0]
                        if all(np.hypot(s[0] - o[0], s[1] - o[1]) > 8.0 for o in stations):
                            break                                    # far enough from the others
                    stations.append(s)
        self.stations = np.array(stations, dtype=np.float32)

        # restaurant (depot): the drone starts here and returns here after every delivery
        for _try in range(50):
            along = rng.uniform(-(WORLD_HALF - 5.0), WORLD_HALF - 5.0)
            road = float(rng.choice(ROAD_CENTERS))
            rp = [road, along] if rng.random() < 0.5 else [along, road]
            if all(np.hypot(rp[0] - s[0], rp[1] - s[1]) > 12.0 for s in self.stations):
                break                                                # keep away from chargers
        self.restaurant = np.array([rp[0], rp[1], 1.5], dtype=np.float32)

        def near_station(x, y):          # keep trees away from chargers AND the restaurant
            pts = np.vstack([self.stations[:, :2], self.restaurant[None, :2]])
            return bool(np.any(np.hypot(pts[:, 0] - x, pts[:, 1] - y) < 4.5))

        # roads (visual only - no collision)
        for c in ROAD_CENTERS:
            self._box([ROAD_W / 2, WORLD_HALF, 0.01], [c, 0, 0.01], [0.2, 0.2, 0.22, 1], False)
            self._box([WORLD_HALF, ROAD_W / 2, 0.01], [0, c, 0.012], [0.2, 0.2, 0.22, 1], False)

        # buildings / parks: each block has 2x2 lots
        for bx in BLOCK_CENTERS:
            for by in BLOCK_CENTERS:
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
            for t in range(-int(WORLD_HALF) + 6, int(WORLD_HALF) - 5, 8):
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

        # restaurant visuals: red pad + pole (visual only)
        rx, ry = float(self.restaurant[0]), float(self.restaurant[1])
        vis = bc.createVisualShape(p.GEOM_CYLINDER, radius=RESTAURANT_RADIUS, length=0.1,
                                   rgbaColor=[0.9, 0.15, 0.15, 1])
        bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[rx, ry, 0.07])
        vis = bc.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=4.0,
                                   rgbaColor=[0.9, 0.15, 0.15, 0.8])
        bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[rx + 1.8, ry, 2.0])
        vis = bc.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[1.0, 1.0, 1.0, 1])
        bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[rx + 1.8, ry, 4.3])

        # parcel carried by the drone (visual only, hidden when not carrying)
        vis = bc.createVisualShape(p.GEOM_BOX, halfExtents=[0.18, 0.18, 0.14],
                                   rgbaColor=[0.6, 0.4, 0.2, 1])
        self.parcel = bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                         basePosition=[0, 0, -10])

        # delivery target marker (visual only, shown while carrying a parcel)
        vis = bc.createVisualShape(p.GEOM_SPHERE, radius=0.6, rgbaColor=[1.0, 0.6, 0.0, 0.8])
        self.marker = bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                         basePosition=[0, 0, 2])

        # the drone
        col = bc.createCollisionShape(p.GEOM_SPHERE, radius=DRONE_R)
        vis = bc.createVisualShape(p.GEOM_SPHERE, radius=DRONE_R, rgbaColor=[0.9, 0.1, 0.1, 1])
        self.drone = bc.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis, basePosition=[0, 0, 2])

    # --------------------------------------------------------------- helpers
    def _sample_road_point(self, z):
        c = self.np_random.choice(ROAD_CENTERS)
        along = self.np_random.uniform(-(WORLD_HALF - 4.0), WORLD_HALF - 4.0)
        off = self.np_random.uniform(-0.6, 0.6)
        if self.np_random.random() < 0.5:
            return np.array([c + off, along, z], np.float32)
        return np.array([along, c + off, z], np.float32)

    def _new_target(self):
        """Next customer. Every trip starts at the restaurant, so distance is measured from it."""
        max_d = 12.0 + (85.0 - 12.0) * self.difficulty   # 12 m (easy) .. 85 m
        best, best_gap = None, 1e9
        for _ in range(100):
            t = self._sample_road_point(self.np_random.uniform(1.5, 3.0))
            d = float(np.linalg.norm(t[:2] - self.restaurant[:2]))
            if 6.0 <= d <= max_d:
                best = t
                break
            gap = (6.0 - d) if d < 6.0 else (d - max_d)
            if gap < best_gap:
                best, best_gap = t, gap
        self.target = best

    def _update_visuals(self):
        hide = [0.0, 0.0, -10.0]
        self.bc.resetBasePositionAndOrientation(
            self.marker, self.target.tolist() if self.carrying else hide, [0, 0, 0, 1])
        parcel_pos = (self.pos + np.array([0, 0, -0.45], np.float32)).tolist()
        self.bc.resetBasePositionAndOrientation(
            self.parcel, parcel_pos if self.carrying else hide, [0, 0, 0, 1])

    def _nearest_station(self):
        return int(np.argmin(np.linalg.norm(self.stations - self.pos, axis=1)))

    def _goal_pos(self):
        if self.seeking:
            return self.stations[self.station_idx]
        if self.carrying:
            return self.target
        return self.restaurant

    def _reserve(self):
        """Battery % needed to reach the nearest charger from here (+ margin)."""
        d = float(np.linalg.norm(self.stations - self.pos, axis=1).min())
        return self.cfg.reserve_base + self.cfg.reserve_per_m * d

    def _trip_energy(self):
        """Battery % for a full trip: restaurant -> next customer -> nearest charger."""
        d_rt = float(np.linalg.norm(self.target[:2] - self.restaurant[:2]))
        d_ts = float(np.linalg.norm(self.stations[:, :2] - self.target[:2], axis=1).min())
        need = self.cfg.trip_per_m * (d_rt + d_ts) + self.cfg.trip_margin
        return min(need, self.cfg.resume_battery - 5.0)

    def _charge_level(self):
        """Battery level below which a drone WITHOUT a parcel must go and charge. If another
        parcel is waiting it must also hold enough energy for the whole trip, because it is
        not allowed to charge while carrying."""
        lvl = self._reserve()
        if self.deliveries < self.n_cur:
            lvl = max(lvl, self._trip_energy())
        return lvl

    def _battery_margin(self):
        """Battery % above (+) or below (-) what the current situation requires."""
        if self.carrying:                       # must finish the delivery, then reach a charger
            d = float(np.linalg.norm(self.target - self.pos))
            d_ts = float(np.linalg.norm(self.stations[:, :2] - self.target[:2], axis=1).min())
            need = self.cfg.trip_per_m * (d + d_ts) + 5.0
        else:
            need = self._charge_level()
        return self.battery - need

    # ------------------------------------------------------------- route guidance
    def _plan_route(self):
        """(Re)plan the road route from the current position to the current goal."""
        goal = self._goal_pos()
        pts = plan_road_route(self.pos[:2], goal[:2], self.blocked)
        wps, lines = [[x, y, CRUISE_Z] for x, y, _ in pts], [ln for _, _, ln in pts]
        wps.append([float(goal[0]), float(goal[1]), float(goal[2])])
        lines.append(pts[-1][2])
        out_w, out_l, prev_xy = [], [], self.pos[:2]
        for idx, (w, ln) in enumerate(zip(wps, lines)):
            last = idx == len(wps) - 1
            if not last and np.linalg.norm(np.array(w[:2], np.float32) - prev_xy) < 0.7:
                continue                                  # already (almost) there
            out_w.append(w); out_l.append(ln)
            prev_xy = np.array(w[:2], np.float32)
        self.route_wps = np.array(out_w, np.float32)
        self.route_lines = out_l
        seg = np.linalg.norm(np.diff(self.route_wps, axis=0), axis=1)
        rem = np.zeros(len(out_w), np.float32)
        if len(seg):
            rem[:-1] = np.cumsum(seg[::-1])[::-1]
        self.route_rem = rem
        self.route_ptr = 0
        self.route_start = self.pos.copy()
        self._last_replan = self.steps
        self._advance_route()
        self.prev_dist = self.best_dist = self._path_remaining()   # no shaping jump on replans
        if self.render_mode == "human":
            self._draw_route()

    def _advance_route(self):
        k, n = self.route_ptr, len(self.route_wps)
        while k < n - 1 and np.linalg.norm(self.route_wps[k][:2] - self.pos[:2]) < WP_RADIUS:
            k += 1
        self.route_ptr = k

    def _path_remaining(self):
        k = self.route_ptr
        return float(np.linalg.norm(self.route_wps[k] - self.pos)) + float(self.route_rem[k])

    def _off_route(self):
        k = self.route_ptr
        a = (self.route_wps[k - 1] if k > 0 else self.route_start)[:2]
        b = self.route_wps[k][:2]
        ab = b - a
        t = float(np.clip(np.dot(self.pos[:2] - a, ab) / max(float(np.dot(ab, ab)), 1e-6), 0.0, 1.0))
        return float(np.linalg.norm(self.pos[:2] - (a + t * ab))) > OFFROUTE_DIST

    def _reroute(self):
        """Stuck: forbid continuing in the current direction along this road segment and plan
        again, so the drone is guided onto a DIFFERENT route (e.g. back to the last junction
        and around another block)."""
        k = self.route_ptr
        line = self.route_lines[k]
        ax = 1 if line[0] == "V" else 0
        delta = float(self.route_wps[k][ax] - self.pos[ax])
        if abs(delta) > 0.1:
            seg = int(np.searchsorted(ROAD_CENTERS, float(self.pos[ax])))
            self.blocked.add((line, seg, 1 if delta > 0 else -1))
        self.reroutes += 1
        self.total_reroutes += 1
        self._plan_route()

    def _route_feats(self):
        k, n = self.route_ptr, len(self.route_wps)
        f1 = self._goal_feat(self.route_wps[k])
        f2 = self._goal_feat(self.route_wps[min(k + 1, n - 1)])
        rem = min(1.0, self._path_remaining() / (4 * WORLD_HALF))
        return np.concatenate([f1, f2, [rem]])

    def _draw_route(self):
        for i in self._dbg_route:
            self.bc.removeUserDebugItem(i)
        color = [1.0, 0.2, 1.0] if self.reroutes else [0.2, 0.8, 1.0]     # magenta = rerouted
        pts = [self.pos] + list(self.route_wps)
        self._dbg_route = [
            self.bc.addUserDebugLine(np.asarray(a).tolist(), np.asarray(b).tolist(),
                                     lineColorRGB=color, lineWidth=3.0)
            for a, b in zip(pts[:-1], pts[1:])]

    def _update_goal(self):
        """Pick the current goal: charger (only when NOT carrying) > customer (carrying) >
        restaurant. Resets the shaping baseline whenever the goal changes, so switching
        goals never pays reward."""
        cfg = self.cfg
        if not self.seeking and not self.carrying and self.battery < self._charge_level():
            self.seeking = True
            self.bonus_given = False
            self.station_idx = self._nearest_station()
        elif self.seeking and self.battery >= cfg.resume_battery:
            self.seeking = False
            self.charges += 1
        if self.seeking:
            key = ("station", self.station_idx)
        elif self.carrying:
            key = ("deliver", self.deliveries)
        else:
            key = ("return", self.deliveries)
        if key != self.goal_key:
            self.goal_key = key
            self.blocked = set()
            self.reroutes = 0
            self._plan_route()                  # also resets prev_dist / best_dist
            self.stall_steps = 0
            self.dwell_steps = 0
            self.dwell_anchor = self.pos.copy()

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
        return np.concatenate([v / max(d, 1e-6), [min(d / (2 * WORLD_HALF), 1.0)]])

    def _obs(self):
        pos = self.pos
        stn = self.stations[self.station_idx] if self.seeking else self.stations[self._nearest_station()]
        obs = np.concatenate([
            self._goal_feat(self.target),
            self._goal_feat(stn),
            self._goal_feat(self.restaurant),
            self._route_feats(),
            [self.battery / 100.0],
            [self._battery_margin() / 50.0],             # >0: enough battery, <0: low
            self.vel / VMAX,
            [pos[0] / WORLD_HALF, pos[1] / WORLD_HALF, pos[2] / MAX_ALT],
            [float(self.seeking)],
            [float(self.carrying)],
            self.lidar / LIDAR_RANGE,
            [min(1.0, self.stall_steps / self.cfg.stall_window)],
            [min(1.0, self.dwell_steps / (self.cfg.dwell_grace + self.cfg.dwell_ramp))],
            self.prev_action,
        ])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ gym
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

        self.pos = self.restaurant.copy()                      # start on the restaurant pad
        self.vel = np.zeros(3, np.float32)
        self.bc.resetBasePositionAndOrientation(self.drone, self.pos.tolist(), [0, 0, 0, 1])
        self.bc.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])

        lo = 80.0 - 45.0 * self.difficulty                       # 80..100% easy, 35..100% full
        self.battery = float(self.np_random.uniform(lo, 100.0))
        if self.difficulty >= 0.2 and self.np_random.random() < self.cfg.low_start_prob:
            self.battery = float(self.np_random.uniform(25.0, 40.0))   # practise charging often
        self.charges = 0
        self.bonus_given = False
        self.n_cur = min(self.n_deliveries,
                         1 + int(self.difficulty * 3 >= 1.0) + int(self.difficulty * 3 >= 2.0))
        self.steps = 0
        self.deliveries = 0
        self.prev_action = np.zeros(3, np.float32)
        self.seeking = False
        self.carrying = False
        self.awaiting_return = False
        self.station_idx = 0
        self.goal_key = None
        self.stall_steps = 0
        self.best_dist = 1e9
        self.blocked = set()
        self.reroutes = 0
        self.total_reroutes = 0
        self.dwell_steps = 0
        self.dwell_anchor = self.pos.copy()
        self._last_replan = -100
        self._new_target()
        self.lidar = self._lidar()
        self._update_goal()                 # battery too low for a trip? -> charge first
        if not self.seeking:
            self.carrying = True            # loads the first parcel
            self._update_goal()
        if self.render_mode == "human":
            self._update_visuals()
        return self._obs(), {}

    def step(self, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)

        # ---- move the drone (smoothed velocity command), check collisions
        self.vel += VEL_ALPHA * (action * VMAX - self.vel)
        if self.seeking:                                    # docking assist: slow down near the charger
            dd = float(np.linalg.norm(self.stations[self.station_idx] - self.pos))
            vlim = max(cfg.dock_min_speed, cfg.dock_gain * dd)
            sp = float(np.linalg.norm(self.vel))
            if sp > vlim:
                self.vel *= vlim / sp
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
        self._advance_route()
        if self.steps - self._last_replan > 20 and self._off_route():
            self._plan_route()                                 # pushed off the route: plan again
        reward = cfg.time_penalty
        dist = self._path_remaining()                          # distance to go ALONG the route
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
        reward -= cfg.limit_coef * limited          # needing the limiter = poor avoidance
        if self.seeking:                            # low battery: every step without charging hurts more
            res = self._reserve()
            reward -= cfg.urgency_coef * float(np.clip((res - self.battery) / res, 0.0, 1.0))

        # ---- hovering in one place: growing penalty -> escape bonus -> re-route
        in_station = (float(np.linalg.norm(self.stations - self.pos, axis=1).min()) < STATION_RADIUS
                      and speed < STATION_MAX_SPEED)
        if in_station and self.seeking:                        # charging is not "stuck"
            self.dwell_anchor = self.pos.copy()
            self.dwell_steps = 0
        elif float(np.linalg.norm(self.pos - self.dwell_anchor)) > cfg.dwell_radius:
            if self.dwell_steps > cfg.dwell_grace:
                reward += cfg.escape_bonus                     # broke out after being stuck
            self.dwell_anchor = self.pos.copy()
            self.dwell_steps = 0
        else:
            self.dwell_steps += 1
        reward -= cfg.dwell_coef * float(
            np.clip((self.dwell_steps - cfg.dwell_grace) / cfg.dwell_ramp, 0.0, 1.0))
        if (self.dwell_steps >= cfg.reroute_after
                and (self.dwell_steps - cfg.reroute_after) % cfg.reroute_every == 0):
            self._reroute()

        # ---- terminal / event logic
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
            if in_station and self.battery < 100.0:
                gain = min(cfg.charge_rate, 100.0 - self.battery)
                if self.seeking:                      # only needed charging pays (anti-farming)
                    reward += cfg.charge_per_pct * gain
                    if not self.bonus_given:
                        reward += cfg.charge_start_bonus
                        self.bonus_given = True
                self.battery += gain

            # delivery: only while carrying a parcel
            if self.carrying and np.linalg.norm(self.target - self.pos) < DELIVERY_RADIUS:
                reward += cfg.delivery
                self.deliveries += 1
                self.carrying = False
                self.awaiting_return = True
                if self.deliveries < self.n_cur:
                    self._new_target()                  # next customer (collected at the restaurant)

            # back at the restaurant (no parcel, not on the way to a charger)
            if ((not self.carrying) and (not self.seeking)
                    and np.linalg.norm(self.restaurant - self.pos) < RESTAURANT_RADIUS):
                if self.awaiting_return:
                    reward += cfg.return_bonus
                    self.awaiting_return = False
                if self.deliveries >= self.n_cur:
                    reward += cfg.all_done_bonus
                    terminated = True; event = "success"
                elif self.battery >= self._charge_level():
                    self.carrying = True                # loads the next parcel

            if not terminated and self.steps >= self.max_steps:
                truncated = True; event = "timeout"

            if not terminated:
                self._update_goal()

        info = {"deliveries": self.deliveries, "battery": self.battery,
                "n_deliveries": self.n_cur, "difficulty": self.difficulty,
                "charges": self.charges, "speed": speed, "carrying": self.carrying,
                "reroutes": self.total_reroutes}
        if event:
            info["event"] = event

        if self.render_mode == "human":
            self.bc.resetDebugVisualizerCamera(38, 45, -40, self.pos.tolist())
            # overlay: line to the CURRENT goal (green = charger, orange = customer, red = restaurant)
            if self.seeking:
                color, mode = [0.1, 0.9, 0.3], "GO CHARGE"
            elif self.carrying:
                color, mode = [1.0, 0.6, 0.0], "DELIVERING PARCEL"
            else:
                color, mode = [0.95, 0.2, 0.2], "RETURN TO RESTAURANT"
            self._dbg_line = self.bc.addUserDebugLine(
                self.pos.tolist(), self.route_wps[self.route_ptr].tolist(), lineColorRGB=color, lineWidth=2.0,
                replaceItemUniqueId=self._dbg_line)
            label = (f"{mode} | battery {self.battery:.0f}% | "
                     f"deliveries {self.deliveries}/{self.n_cur}"
                     + (f" | REROUTED x{self.reroutes}" if self.reroutes else ""))
            self._dbg_text = self.bc.addUserDebugText(
                label, (self.pos + np.array([0, 0, 1.6], np.float32)).tolist(),
                textColorRGB=[1, 1, 1], textSize=1.5, replaceItemUniqueId=self._dbg_text)
            self._dbg_speed = self.bc.addUserDebugText(
                f"SPEED {speed:.1f} m/s ({speed * 3.6:.0f} km/h)",
                (self.pos + np.array([0, 0, 2.6], np.float32)).tolist(),
                textColorRGB=[1.0, 1.0, 0.2], textSize=2.0, replaceItemUniqueId=self._dbg_speed)
            self._update_visuals()
            time.sleep(1.0 / CONTROL_HZ)

        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.bc.disconnect()
        except Exception:
            pass


# ------------------------------------------------------------ quick self-test
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
