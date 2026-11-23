"""
city_env.py  -  PyBullet city delivery-drone environment (Gymnasium API)

World: N_BLOCKS x N_BLOCKS blocks of buildings/parks (4 -> 80 x 80 m), roads between blocks,
trees along roads. Charging stations are buildings too (green, door, sign, the text CHARGING STATION):
one per quadrant, spread out (>= STATION_MIN_GAP apart) and away from the map border. The drone has
to come up to a station's door to charge.
Restaurant: ONE building per city is the restaurant (warm colour, one window, a door, a sign and
the text "RESTAURANT" in GUI mode). The drone starts in front of its door and returns there.
Customers: deliveries go to real buildings - the target is a spot ~1.2 m in front of a
road-facing wall, so the drone stops short of the building instead of flying to a road point.
Flat pads mark the exact spots: red = restaurant, green = charging station.
Other buildings get floor-by-floor windows (visual only, GUI mode only by default).
Drone: point-mass sphere controlled by velocity commands (real PyBullet
collisions with buildings, trees and ground).
GUI-only extras (all visual, none of them affect physics, lidar or rewards): drone body with
rotors, order labels, dropped parcel, intro camera zoom, building windows.

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
STATIONS_PER_QUADRANT = 1
N_STATIONS = 4 * STATIONS_PER_QUADRANT
MAX_ALT = 30.0                     # ceiling (tallest building is 22 m)
ROAD_W = 6.0


def _is_road(c):
    """True if a road centre line runs at coordinate c."""
    return any(abs(c - r) < 1e-3 for r in ROAD_CENTERS)


# ---- road-network route planner ------------------------------------------------------------------------------
# Every start/goal lies next to a road, and roads are free corridors between the buildings, so
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

    # 1) graph: every road is a line with a stop at each junction, plus the start (S) and goal (G)

    n = len(ROAD_CENTERS)
    stops = {line: [] for line in ROAD_LINES}
    for i in range(n):                                    # V road i crosses H road j at (R[i], R[j])
        for j in range(n):
            node = ("I", i, j)
            stops[("V", i)].append((ROAD_CENTERS[j], node))
            stops[("H", j)].append((ROAD_CENTERS[i], node))
    stops[s_line].append((s_c, "S"))
    stops[g_line].append((g_c, "G"))

    # 2) connect neighbouring stops along each road, both directions (cost = length, + BLOCK_COST if blocked)

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

    # 3) Dijkstra from S to G (cnt is a tie-breaker so the heap never has to compare nodes)

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

    # 4) walk back from G to S to recover the path

    nodes, lines, u = [], [], "G"
    while u != "S":
        pu, line = prev[u]
        nodes.append(u); lines.append(line)
        u = pu
    nodes.append("S"); lines.append(s_line)
    nodes.reverse(); lines.reverse()

    # 5) keep only the corner points (drop points in the middle of a straight stretch)

    out = []
    for k, (node, line) in enumerate(zip(nodes, lines)):
        if 0 < k < len(nodes) - 1 and lines[k] == lines[k + 1]:
            continue                                      # straight through: not a corner
        x, y = xy_of(node)
        out.append((float(x), float(y), line))
    return out


# ---- charging, delivery and physics constants--------------------------------------------------------------
STATION_RADIUS = 1.5               # must be within this (3D) of the door spot to charge
STATION_MAX_SPEED = 3.5            # ...and slower than this
SAFETY_STOP_TIME = 0.2             # s; used by the collision-avoidance speed limiter
DELIVERY_RADIUS = 1.5              # within this (3D) of the delivery spot = parcel delivered
RESTAURANT_RADIUS = 2.5            # within this (3D) of the pad = at the restaurant

VMAX = 6.0                         # m/s
VEL_ALPHA = 0.25                   # velocity smoothing (fake inertia)
SUBSTEPS = 12                      # 240 Hz physics / 12 = 20 Hz control
CONTROL_HZ = 20
LIDAR_RANGE = 12.0                 # m; rays report this when nothing is hit
DRONE_R = 0.3                      # drone collision radius (m)

# ---- restaurant / customer buildings
DOOR_GAP = 1.2                     # restaurant / charger spot: this far out from the wall (pad sits at the door)
TARGET_STANDOFF = 1.2              # delivery spot: this far out from the customer's wall
TREE_CLEAR = 2.2                   # delivery spots keep this horizontal distance from tree trunks
FLOOR_H = 3.0                      # window rows are one floor apart
RESTAURANT_MAX_H = 6.0             # the restaurant is a small building
STATION_MAX_H = 6.0                # ...and so is a charging station
STATION_MIN_GAP = 30.0             # stations are at least this far apart (road spots, m)
STATION_ALONG_MIN = 10.0           # ...and at least this far from the map centre lines
STATION_EDGE_MARGIN = 12.0         # ...and at least this far from the map border (along the road)


@dataclass
class RewardConfig:
    """All reward, battery and anti-stuck tuning numbers in one place (train.py reads gamma from here)."""
    # --- per-step terms
    time_penalty: float = -0.05      # every step costs a little
    progress_coef: float = 0.3       # reward per metre of ROUTE progress
    gamma: float = 0.995             # single source of truth: train.py reads this for PPO
    shaping_gamma: float = 1.0       # 1.0 = pure progress reward: total shaping telescopes to
                                     # coef*(start_dist - end_dist), so hovering earns nothing
    smooth_coef: float = 0.08        # penalises jittery actions (change between steps)
    prox_coef: float = 0.3           # penalises flying too close to obstacles
    safe_dist: float = 2.0           # m: closer than this to an obstacle (any lidar ray) starts to cost reward
    safe_down: float = 1.0           # m: same for the downward ray (ground)
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
    reroute_every: int = 60          # ...and again every this many steps while still stuck
    # --- events (one-off rewards and penalties)
    delivery: float = 50.0           # parcel delivered
    all_done_bonus: float = 30.0     # all parcels delivered AND drone back at the restaurant
    return_bonus: float = 10.0       # back at the restaurant after a delivery
    trip_per_m: float = 0.6          # battery % per metre assumed when planning a delivery trip
    trip_margin: float = 8.0         # extra % kept in reserve for a trip
    charge_per_pct: float = 0.5      # per % gained, ONLY while in seek-charger mode (anti-farming)
    crash: float = -100.0            # episode ends
    battery_dead: float = -100.0     # battery hit 0 (episode ends)
    out_of_bounds: float = -100.0    # left the map or flew above MAX_ALT (episode ends)
    # --- battery model (percent per control step)
    drain_base: float = 0.04         # % per step just for being airborne
    drain_speed: float = 0.10        # extra % per step at full speed (grows with speed squared)
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
    """Food-delivery drone. Each episode: carry up to n_deliveries parcels one at a time from the
    restaurant to customer buildings and fly back to the restaurant, charging at a station when
    the battery is low. A random new city is generated per episode (or one fixed city)."""

    metadata = {"render_modes": ["human"], "render_fps": CONTROL_HZ}

    def __init__(self, render_mode=None, layout_seed=None, layout_pool=(0, 1000),
                 regen_every=1, n_deliveries=3, max_steps=2000, reward_cfg=None,
                 difficulty=1.0, difficulty_mix=False, safety_filter=True,
                 detail_buildings=None, follow_camera=True):
        """
        layout_seed : int -> one fixed city forever.
                      None -> a NEW random city is generated during reset(), drawn
                              from layout_pool = (low, high) seeds.
        regen_every : with layout_seed=None, rebuild the city every N episodes.
        detail_buildings : floor-by-floor windows on the buildings (visual only, no collision).
                      None -> only when render_mode == "human" (keeps training fast).
                      True/False -> force on/off.
        follow_camera : GUI only. True -> the camera follows the drone every step.
                      False -> the camera is placed once at the first reset and then left alone,
                      so you can move it freely with the mouse (Ctrl + drag, scroll to zoom).
        Keep train pools and evaluation pools disjoint (e.g. 0-999 vs 10000-10099).
        """
        super().__init__()
        self.render_mode = render_mode
        self.follow_camera = bool(follow_camera)
        self._cam_set = False
        self.detail_buildings = (render_mode == "human") if detail_buildings is None else bool(detail_buildings)
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
        self.drop_steps = 0                       # >0 while the dropped parcel is shown at the customer
        self.drop_pos = np.zeros(3, np.float32)
        self.dropped = -1                         # body id of the dropped parcel (visual only)
        self._rotors = []                         # visual-only drone parts: arms + rotor discs (GUI only)
        self._intro_steps = 0                     # GUI: length of the city-overview zoom-in at the start

        # action = desired velocity (x, y, z) in [-1, 1], scaled by VMAX

        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

        # target(4) + station(4) + restaurant(4) + route(9) + battery1 + battery-margin1 + vel3 + pos3
        # + seeking1 + carrying1 + lidar26 + stall timer1 + dwell timer1 + previous action3 = 62

        self.observation_space = spaces.Box(-1.0, 1.0, (62,), np.float32)

        # GUI opens the PyBullet window; DIRECT runs without one (training)

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

        # world state, filled in by _build_world() / reset()
        self.stations = np.zeros((N_STATIONS, 3), np.float32)   # charging spots (in front of each station door)
        self.restaurant = np.zeros(3, np.float32)                # start / return spot (in front of the restaurant door)
        self.carrying = False
        self.awaiting_return = False
        self.pos = np.zeros(3, np.float32)
        self.vel = np.zeros(3, np.float32)
        self.target = np.zeros(3, np.float32)         # delivery spot, in front of the customer's wall
        self.target_marker = np.zeros(3, np.float32)  # where the marker is drawn (on the facade)
        self.tree_xy = []
        self._tree_arr = np.zeros((0, 2), np.float32)
        self.faces = []                               # road-facing walls of customer buildings
        self._win_vis = {}
        self._deco_rng = np.random.default_rng(0)
        self.lidar = np.full(26, LIDAR_RANGE, np.float32)

    def set_difficulty(self, d):
        """Curriculum hook (called from train.py through env_method). Takes effect
        at the next reset(): d=0 -> sparse low city, close targets, 1 delivery, full
        battery. d=1 -> the full task."""
        self.level = float(np.clip(d, 0.0, 1.0))
        self.difficulty = self.level

    # ------------------------------------------------------------ world build------------------------------
    def _box(self, half, pos, color, collide=True):
        """Create a static (mass 0) box. collide=False makes it visual-only: it has no collision shape, so
        the lidar, crash detection and the safety limiter all ignore it."""
        col = self.bc.createCollisionShape(p.GEOM_BOX, halfExtents=half) if collide else -1
        vis = self.bc.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        return self.bc.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                       baseVisualShapeIndex=vis, basePosition=pos)

    def _tree(self, x, y):
        """A tree = trunk (cylinder) + canopy (sphere), both with collision. The position is stored in
        self.tree_xy so delivery spots can keep away from trunks."""
        bc = self.bc
        self.tree_xy.append((float(x), float(y)))
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

    def _windows(self, cx, cy, hx, hy, h):

        """Floor-by-floor windows on all four walls of a building. Visual only (no collision
        shape), so lidar, crash detection and the safety limiter are unaffected. Uses its own
        random generator, so the seeded city layout never changes."""
        rng, bc = self._deco_rng, self.bc
        zs = np.arange(1.8, h - 1.0 + 1e-6, FLOOR_H)          # one row per floor, gap at the bottom
        for axis, sgn in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)):
            half_along = hy if axis == 0 else hx
            n = max(1, int((2 * half_along - 0.8) // 1.5))    # windows per row on this wall
            offs = (np.arange(n) - (n - 1) / 2.0) * 1.5
            for z in zs:
                for o in offs:
                    u = rng.random()
                    kind = "lit" if u < 0.2 else ("blue" if u < 0.6 else "dark")
                    if axis == 0:
                        pos = [cx + sgn * (hx + 0.02), cy + float(o), float(z)]
                    else:
                        pos = [cx + float(o), cy + sgn * (hy + 0.02), float(z)]
                    bc.createMultiBody(baseMass=0, baseVisualShapeIndex=self._win_vis[(axis, kind)],
                                       basePosition=pos)

    def _door(self, cx, cy, hx, hy, rp, vert, road):

        """Door of a lot facing the road at `road` -> (axis, sign, door coord along the wall,
        door spot xy = DOOR_GAP out from the wall)."""
        if vert:                                              # door faces a vertical road
            axis, sgn = 0, (1.0 if road > cx else -1.0)
            along_w = float(np.clip(rp[1], cy - hy + 1.0, cy + hy - 1.0))
            return axis, sgn, along_w, [cx + sgn * (hx + DOOR_GAP), along_w]
        axis, sgn = 1, (1.0 if road > cy else -1.0)           # door faces a horizontal road
        along_w = float(np.clip(rp[0], cx - hx + 1.0, cx + hx - 1.0))
        return axis, sgn, along_w, [along_w, cy + sgn * (hy + DOOR_GAP)]

    def _building_details(self, info, text, text_color, sign_color, roof_color, door_color, window_color):
        """Door, ONE window, sign board, roof cap and (GUI only) a floating name text, for the
        restaurant and the charging stations. Everything is visual only. Built in both GUI and
        training mode (a handful of bodies per building)."""
        cx, cy, hx, hy, h, axis, sgn, along = info
        half_along = hy if axis == 0 else hx
        d = along - (cy if axis == 0 else cx)                 # door offset along the wall

        def on_wall(off, z, w_half, z_half, color, out=0.03):
            if axis == 0:
                self._box([0.03, w_half, z_half],
                          [cx + sgn * (hx + out), cy + off, z], color, collide=False)
            else:
                self._box([w_half, 0.03, z_half],
                          [cx + off, cy + sgn * (hy + out), z], color, collide=False)

        on_wall(d, 1.1, 0.55, 1.1, door_color)                 # door
        left = d + half_along - 1.05                           # free wall length left / right of door
        right = half_along - d - 1.05
        if left >= right:
            w_off, w_len = (-half_along + 0.3 + d - 0.75) / 2.0, left
        else:
            w_off, w_len = (d + 0.75 + half_along - 0.3) / 2.0, right
        on_wall(w_off, 2.4, min(0.7, w_len / 2.0), 0.7, window_color)               # the one window
        on_wall(0.0, h - 0.7, min(1.6, half_along - 0.2), 0.3, sign_color, out=0.05)  # sign board
        self._box([hx + 0.2, hy + 0.2, 0.15], [cx, cy, h + 0.15], roof_color, collide=False)  # roof cap
        if self.render_mode == "human":
            self.bc.addUserDebugText(text, [cx, cy, h + 1.4], textColorRGB=list(text_color[:3]),
                                     textSize=1.5)

    def _build_world(self, layout_seed):

        """Build one complete city from `layout_seed`. The order fixes the random draws, so a seed always
        gives the same city: style -> charging stations -> restaurant -> roads -> buildings and parks
        (restaurant and station lots become special buildings) -> trees -> pads and labels -> parcel,
        marker and drone."""

        bc = self.bc
        bc.resetSimulation()                 # wipe the previous city
        if self.render_mode == "human":
            bc.removeAllUserDebugItems()
        self._dbg_line, self._dbg_text, self._dbg_speed = -1, -1, -1
        self._dbg_route = []
        self.tree_xy = []
        self.faces = []
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        bc.setGravity(0, 0, 0)               # velocity-controlled drone, no gravity
        bc.setTimeStep(1.0 / 240.0)
        rng = np.random.default_rng(layout_seed)
        # separate generators: they never touch `rng`, so a given seed gives the same city as before
        self._deco_rng = np.random.default_rng([int(layout_seed), 1])    # window colours
        forced_rng = np.random.default_rng([int(layout_seed), 2])        # size of forced buildings
        self._built_seed = layout_seed
        # per-city style: park density and how tall the skyline is
        park_p = rng.uniform(0.05, 0.30)
        max_h = rng.uniform(12.0, 22.0)
        empty_p = 0.6 * (1.0 - self.difficulty)       # easy levels: many empty lots

        plane = bc.loadURDF("plane.urdf")
        bc.changeVisualShape(plane, -1, rgbaColor=[0.35, 0.55, 0.3, 1], textureUniqueId=-1)

        # shared window shapes (created once per city, reused by every window)
        self._win_vis = {}
        if self.detail_buildings:
            cols = {"dark": [0.10, 0.14, 0.22, 1], "blue": [0.22, 0.34, 0.50, 1],
                    "lit": [1.0, 0.85, 0.40, 1]}
            for name, col in cols.items():
                self._win_vis[(0, name)] = bc.createVisualShape(
                    p.GEOM_BOX, halfExtents=[0.03, 0.4, 0.5], rgbaColor=col)
                self._win_vis[(1, name)] = bc.createVisualShape(
                    p.GEOM_BOX, halfExtents=[0.4, 0.03, 0.5], rgbaColor=col)

        # charging stations: STATIONS_PER_QUADRANT per quadrant, on the quadrant's own (outer) road
        # lines, STATION_EDGE_MARGIN away from the map border and >= STATION_MIN_GAP from each other.
        # Each station is a BUILDING (nearest free lot); the charging spot is in front of its door
        # and is set once the building is made. `spots` are the road points picked here.

        lot_list = [(bx + dx, by + dy) for bx in BLOCK_CENTERS for by in BLOCK_CENTERS
                    for dx in (-3.5, 3.5) for dy in (-3.5, 3.5)]
        taken = set()                                        # lots already given to a station/restaurant

        def pick_lot(rp_, vert_, road_):
            cands = [l for l in lot_list if l not in taken
                     and abs((l[0] if vert_ else l[1]) - road_) < 8.0]
            lot_ = min(cands, key=lambda l: float(np.hypot(l[0] - rp_[0], l[1] - rp_[1])))
            taken.add(lot_)
            return lot_

        spots = []
        for qx in (-1, 1):
            for qy in (-1, 1):
                for _ in range(STATIONS_PER_QUADRANT):
                    best, best_gap = None, -1.0
                    for _try in range(30):
                        along = rng.uniform(STATION_ALONG_MIN, WORLD_HALF - STATION_EDGE_MARGIN)
                        vert = bool(rng.random() < 0.5)
                        if vert:                             # on a vertical road of this quadrant
                            road = float(rng.choice([r for r in ROAD_CENTERS if np.sign(r) == qx]))
                            rp_ = [road, qy * along]
                        else:                                # on a horizontal road of this quadrant
                            road = float(rng.choice([r for r in ROAD_CENTERS if np.sign(r) == qy]))
                            rp_ = [qx * along, road]
                        gap = min((float(np.hypot(rp_[0] - o[0][0], rp_[1] - o[0][1])) for o in spots),
                                  default=1e9)
                        if gap >= STATION_MIN_GAP:           # far enough from the others: take it
                            best = (rp_, vert, road)
                            break
                        if gap > best_gap:                   # else remember the farthest candidate
                            best, best_gap = (rp_, vert, road), gap
                    spots.append(best)
        special = {}                                         # lot -> (kind, index, rp, vert, road)
        for i, (rp_, vert, road) in enumerate(spots):
            special[pick_lot(rp_, vert, road)] = ("station", i, rp_, vert, road)
        self.stations = np.array([[s_[0][0], s_[0][1], 1.5] for s_ in spots], dtype=np.float32)  # provisional
        station_doors = [None] * len(spots)
        station_infos = [None] * len(spots)

        # restaurant road spot (same random draws as before). The restaurant BUILDING is the free lot
        # nearest to this spot; the drone start/pad is placed at its door once the lot is built.

        for _try in range(50):
            along = rng.uniform(-(WORLD_HALF - 5.0), WORLD_HALF - 5.0)
            road = float(rng.choice(ROAD_CENTERS))
            rest_vert = bool(rng.random() < 0.5)
            rp = [road, along] if rest_vert else [along, road]
            if all(np.hypot(rp[0] - s[0], rp[1] - s[1]) > 12.0 for s in self.stations):
                break                                                # keep away from chargers
        self.restaurant = np.array([rp[0], rp[1], 1.5], dtype=np.float32)   # provisional
        rest_lot = pick_lot(rp, rest_vert, road)
        special[rest_lot] = ("restaurant", 0, rp, rest_vert, road)
        rest_info = None

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
                        role = special.get((cx, cy))        # restaurant / charging station lot?
                        if r < empty_p:
                            kind = "empty"                  # empty lot (easy levels only)
                        elif r < empty_p + park_p:
                            kind = "park"
                        else:
                            kind = "building"
                        # restaurant and station lots are always buildings; on easy levels empty lots
                        # near the restaurant are filled so there are customers close by
                        near_rest = float(np.hypot(cx - rest_lot[0], cy - rest_lot[1])) < 15.0
                        forced = role is not None or (kind == "empty" and near_rest)

                        if kind == "building":
                            h = rng.uniform(5.0, 6.0 + (max_h - 6.0) * self.difficulty)
                            half = [rng.uniform(2.2, 3.0), rng.uniform(2.2, 3.0), h / 2]
                            shade = rng.uniform(0.45, 0.75)
                        elif forced:                        # does not touch `rng`
                            h = forced_rng.uniform(5.0, 6.0)
                            half = [forced_rng.uniform(2.2, 3.0), forced_rng.uniform(2.2, 3.0), h / 2]
                            shade = forced_rng.uniform(0.45, 0.75)
                        else:
                            if kind == "park":
                                self._tree(cx, cy)
                            continue

                        if role is not None:
                            h = min(h, RESTAURANT_MAX_H if role[0] == "restaurant" else STATION_MAX_H)
                            half[2] = h / 2                 # small building
                            hx, hy = half[0], half[1]
                            axis, sgn, along_w, door = self._door(cx, cy, hx, hy, role[2], role[3], role[4])
                            info = (cx, cy, hx, hy, h, axis, sgn, along_w)
                            if role[0] == "restaurant":
                                self.restaurant = np.array([door[0], door[1], 1.5], dtype=np.float32)
                                rest_info = info
                                self._box(half, [cx, cy, h / 2], [0.92, 0.50, 0.18, 1])   # warm orange
                            else:
                                station_doors[role[1]] = [door[0], door[1], 1.5]
                                station_infos[role[1]] = info
                                self._box(half, [cx, cy, h / 2], [0.15, 0.65, 0.35, 1])   # green
                            continue

                        self._box(half, [cx, cy, h / 2], [shade, shade, shade + 0.1, 1])
                        if self.detail_buildings:
                            self._windows(cx, cy, half[0], half[1], h)
                        # road-facing walls = places a parcel can be delivered
                        sx, sy = (1.0 if dx > 0 else -1.0), (1.0 if dy > 0 else -1.0)
                        if _is_road(bx + sx * PITCH / 2):
                            self.faces.append((cx, cy, half[0], half[1], 0, sx))
                        if _is_road(by + sy * PITCH / 2):
                            self.faces.append((cx, cy, half[0], half[1], 1, sy))

        # the charging spots are now the stations' door spots (trees keep away from them)
        self.stations = np.array(station_doors, dtype=np.float32)

        # trees along roads (away from intersections)
        for c in ROAD_CENTERS:
            for t in range(-int(WORLD_HALF) + 6, int(WORLD_HALF) - 5, 8):
                if any(abs(t - r) < 6 for r in ROAD_CENTERS):
                    continue
                for tx, ty in ((c + rng.choice([-1, 1]) * 2.8, t),
                               (t, c + rng.choice([-1, 1]) * 2.8)):
                    if not near_station(tx, ty):
                        self._tree(tx, ty)
        self._tree_arr = np.array(self.tree_xy, np.float32).reshape(-1, 2)

        # charging stations: building details (sign + text) + flat charging pad in front of the
        # door (charging itself is handled in step(); the pad is drawn at the charging radius)

        for st, info in zip(self.stations, station_infos):
            self._building_details(info, "CHARGING STATION", [0.3, 1.0, 0.5], [1.0, 0.85, 0.1, 1],
                                   [0.05, 0.30, 0.13, 1], [0.05, 0.20, 0.10, 1], [0.70, 1.0, 0.90, 1])
            vis = bc.createVisualShape(p.GEOM_CYLINDER, radius=STATION_RADIUS, length=0.1,
                                       rgbaColor=[0.1, 0.9, 0.3, 1])
            bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                               basePosition=[st[0], st[1], 0.06])

        # restaurant: building details + flat landing pad in front of the door
        self._building_details(rest_info, "RESTAURANT", [1.0, 0.9, 0.2], [0.85, 0.1, 0.1, 1],
                               [0.55, 0.1, 0.1, 1], [0.25, 0.14, 0.07, 1], [1.0, 0.9, 0.55, 1])
        rx, ry = float(self.restaurant[0]), float(self.restaurant[1])
        vis = bc.createVisualShape(p.GEOM_CYLINDER, radius=RESTAURANT_RADIUS, length=0.1,
                                   rgbaColor=[0.9, 0.15, 0.15, 1])
        bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[rx, ry, 0.07])

        # parcel carried by the drone (visual only, hidden when not carrying)
        vis = bc.createVisualShape(p.GEOM_BOX, halfExtents=[0.18, 0.18, 0.14],
                                   rgbaColor=[0.6, 0.4, 0.2, 1])
        self.parcel = bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                         basePosition=[0, 0, -10])

        # delivery target marker (visual only, shown while carrying a parcel)
        vis = bc.createVisualShape(p.GEOM_SPHERE, radius=0.6, rgbaColor=[1.0, 0.6, 0.0, 0.8])
        self.marker = bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                         basePosition=[0, 0, 2])

        # parcel left at the customer after a delivery (visual only, shown briefly)
        vis = bc.createVisualShape(p.GEOM_BOX, halfExtents=[0.18, 0.18, 0.14],
                                   rgbaColor=[0.6, 0.4, 0.2, 1])
        self.dropped = bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                          basePosition=[0, 0, -10])

        # the drone: red body with the real collision sphere. Arms + rotor discs are separate
        # visual-only bodies (GUI only) that follow the drone, so physics and lidar are unchanged.
        col = bc.createCollisionShape(p.GEOM_SPHERE, radius=DRONE_R)
        vis = bc.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.07], rgbaColor=[0.9, 0.1, 0.1, 1])
        self.drone = bc.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis, basePosition=[0, 0, 2])
        self._rotors = []
        if self.render_mode == "human":
            arm_x = bc.createVisualShape(p.GEOM_BOX, halfExtents=[0.36, 0.03, 0.015],
                                         rgbaColor=[0.15, 0.15, 0.15, 1])
            arm_y = bc.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.36, 0.015],
                                         rgbaColor=[0.15, 0.15, 0.15, 1])
            disc = bc.createVisualShape(p.GEOM_CYLINDER, radius=0.17, length=0.012,
                                        rgbaColor=[0.75, 0.80, 0.85, 0.55])
            parts = [(arm_x, (0, 0, 0.03)), (arm_y, (0, 0, 0.03)),
                     (disc, (0.36, 0, 0.07)), (disc, (-0.36, 0, 0.07)),
                     (disc, (0, 0.36, 0.07)), (disc, (0, -0.36, 0.07))]
            self._rotors = [(bc.createMultiBody(baseMass=0, baseVisualShapeIndex=v,
                                                basePosition=[0, 0, -10]),
                             np.array(o, np.float32)) for v, o in parts]

    # --------------------------------------------------------------- helpers-------------
    def _sample_road_point(self, z):
        """Random point on a road centre line at height z (fallback delivery target if a city has no usable building)."""
        c = self.np_random.choice(ROAD_CENTERS)
        along = self.np_random.uniform(-(WORLD_HALF - 4.0), WORLD_HALF - 4.0)
        off = self.np_random.uniform(-0.6, 0.6)
        if self.np_random.random() < 0.5:
            return np.array([c + off, along, z], np.float32)
        return np.array([along, c + off, z], np.float32)

    def _near_tree(self, x, y):
        """True if (x, y) is closer than TREE_CLEAR to any tree trunk."""
        if len(self._tree_arr) == 0:
            return False
        return float(np.hypot(self._tree_arr[:, 0] - x, self._tree_arr[:, 1] - y).min()) < TREE_CLEAR

    def _new_target(self):
        """Next customer = a real building. The delivery spot is TARGET_STANDOFF in front of a
        road-facing wall (the drone stops short of the building); the marker sits on the facade.
        Every trip starts at the restaurant, so distance is measured from it."""
        max_d = 12.0 + (85.0 - 12.0) * self.difficulty   # 12 m (easy) .. 85 m
        best, best_gap = None, 1e9
        if self.faces:
            for _ in range(100):
                cx, cy, hx, hy, axis, sgn = self.faces[int(self.np_random.integers(len(self.faces)))]
                half_along = hy if axis == 0 else hx
                off = float(self.np_random.uniform(-(half_along - 0.8), half_along - 0.8))
                z = float(self.np_random.uniform(1.5, 3.0))
                if axis == 0:
                    wx, wy = cx + sgn * hx, cy + off
                    stand = (wx + sgn * TARGET_STANDOFF, wy)
                    mark = (wx + sgn * 0.3, wy)
                else:
                    wx, wy = cx + off, cy + sgn * hy
                    stand = (wx, wy + sgn * TARGET_STANDOFF)
                    mark = (wx, wy + sgn * 0.3)
                if self._near_tree(stand[0], stand[1]):
                    continue
                d = float(np.hypot(stand[0] - self.restaurant[0], stand[1] - self.restaurant[1]))
                cand = (np.array([stand[0], stand[1], z], np.float32),
                        np.array([mark[0], mark[1], z], np.float32))
                if 6.0 <= d <= max_d:
                    best = cand
                    break
                gap = (6.0 - d) if d < 6.0 else (d - max_d)
                if gap < best_gap:
                    best, best_gap = cand, gap
        if best is None:                                  # no usable building: old road-point target
            for _ in range(100):
                t = self._sample_road_point(self.np_random.uniform(1.5, 3.0))
                d = float(np.linalg.norm(t[:2] - self.restaurant[:2]))
                if 6.0 <= d <= max_d:
                    best = (t, t.copy())
                    break
                gap = (6.0 - d) if d < 6.0 else (d - max_d)
                if gap < best_gap:
                    best, best_gap = (t, t.copy()), gap
        self.target, self.target_marker = best

    def _update_visuals(self):
        """GUI only: move the visual-only objects (customer marker, carried parcel, dropped parcel, rotor
        parts) to where they belong. Hidden objects are parked below the ground."""
        hide = [0.0, 0.0, -10.0]
        self.bc.resetBasePositionAndOrientation(
            self.marker, self.target_marker.tolist() if self.carrying else hide, [0, 0, 0, 1])
        parcel_pos = (self.pos + np.array([0, 0, -0.45], np.float32)).tolist()
        self.bc.resetBasePositionAndOrientation(
            self.parcel, parcel_pos if self.carrying else hide, [0, 0, 0, 1])
        self.bc.resetBasePositionAndOrientation(              # parcel left at the customer's spot
            self.dropped, self.drop_pos.tolist() if self.drop_steps > 0 else hide, [0, 0, 0, 1])
        for body, off in self._rotors:                        # arms + rotor discs follow the drone
            self.bc.resetBasePositionAndOrientation(body, (self.pos + off).tolist(), [0, 0, 0, 1])

    def _nearest_station(self):
        """Index of the charging station closest to the drone."""
        return int(np.argmin(np.linalg.norm(self.stations - self.pos, axis=1)))

    def _goal_pos(self):
        """Where the drone is heading now: charger (seeking) > customer (carrying) > restaurant."""
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
        # route corners at cruise height, then the goal itself as the last waypoint
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
        # route_rem[k] = path length from waypoint k to the end (the progress reward uses it)
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
        """Move the route pointer past every waypoint the drone has reached (horizontal distance < WP_RADIUS)."""
        k, n = self.route_ptr, len(self.route_wps)
        while k < n - 1 and np.linalg.norm(self.route_wps[k][:2] - self.pos[:2]) < WP_RADIUS:
            k += 1
        self.route_ptr = k

    def _path_remaining(self):
        """Distance still to fly along the route: to the current waypoint plus the rest of the route. The
        progress reward is the decrease of this number."""
        k = self.route_ptr
        return float(np.linalg.norm(self.route_wps[k] - self.pos)) + float(self.route_rem[k])

    def _off_route(self):
        """True if the drone is more than OFFROUTE_DIST from the current route leg (pushed or drifted
        away), which makes step() plan a new route."""
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
        """Route part of the observation (9 numbers): direction + distance to the current and the next
        waypoint, and the fraction of the route still to fly."""
        k, n = self.route_ptr, len(self.route_wps)
        f1 = self._goal_feat(self.route_wps[k])
        f2 = self._goal_feat(self.route_wps[min(k + 1, n - 1)])
        rem = min(1.0, self._path_remaining() / (4 * WORLD_HALF))
        return np.concatenate([f1, f2, [rem]])

    def _draw_route(self):
        """GUI only: draw the planned route (cyan; magenta once it has been rerouted)."""
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
        """Cast the 26 rays and return the distance to the first obstacle on each (LIDAR_RANGE if nothing
        is hit). Only shapes with a collision shape are hit."""
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
        """Unit direction (full-strength signal at any range) + distance / (2 * WORLD_HALF), capped at 1."""
        v = goal - self.pos
        d = float(np.linalg.norm(v))
        return np.concatenate([v / max(d, 1e-6), [min(d / (2 * WORLD_HALF), 1.0)]])

    def _obs(self):
        """Build the 62-number observation, all scaled to [-1, 1]. Order: customer spot (4), charger (4),
        restaurant (4), route (9), battery (1), battery margin (1), velocity (3), position (3),
        seeking flag (1), carrying flag (1), lidar (26), stall timer (1), dwell timer (1), previous
        action (3). Each goal block = unit direction (3) + distance (1)."""
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
        """Start a new episode: build a new city if needed, put the drone at the restaurant door, pick the
        starting battery and the first customer, and load the first parcel (or go charging first if
        the battery is too low for a trip)."""
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

        self.pos = self.restaurant.copy()                      # start in front of the restaurant door
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
        self.drop_steps = 0
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
        self._intro_steps = 60 if (self.render_mode == "human" and self._episodes == 1) else 0
        if self.render_mode == "human":
            self._update_visuals()
            if not self.follow_camera and not self._cam_set:
                self.bc.resetDebugVisualizerCamera(60, 45, -40, [0, 0, 0])   # whole-city view, once
                self._cam_set = True
            elif self.follow_camera and self._intro_steps:
                self.bc.resetDebugVisualizerCamera(80, 45, -40, [0, 0, 0])   # start with a wide city view
        return self._obs(), {}

    def step(self, action):

        """One control step (20 Hz): move the drone (velocity command -> docking assist -> safety limiter ->
        physics), drain the battery, compute the reward terms, then handle the episode events (crash,
        battery dead, out of bounds, stall, charging, delivery, return to the restaurant, timeout).
        Returns (obs, reward, terminated, truncated, info)."""
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
        if self.drop_steps > 0:
            self.drop_steps -= 1
        speed = float(np.linalg.norm(self.vel))
        self.lidar = self._lidar()

        # ---- battery drain----------
        self.battery = max(0.0, self.battery - (cfg.drain_base + cfg.drain_speed * (speed / VMAX) ** 2))

        # ---- dense reward terms-----------
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

        # ---- hovering in one place: growing penalty -> escape bonus -> re-route---------------
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

        # ---- terminal / event logic---------------------
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

            # delivery: only while carrying a parcel (spot in front of the customer's wall)
            if self.carrying and np.linalg.norm(self.target - self.pos) < DELIVERY_RADIUS:
                reward += cfg.delivery
                self.deliveries += 1
                self.drop_pos = np.array([self.target[0], self.target[1], 0.14], np.float32)
                self.drop_steps = 60                    # parcel stays at the customer's spot for 3 s (GUI)
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
            if self.follow_camera:
                # first episode: zoom from the wide city view onto the drone, then follow it
                t = min(1.0, self.steps / self._intro_steps) if self._intro_steps else 1.0
                self.bc.resetDebugVisualizerCamera(80.0 * (1 - t) + 38.0 * t, 45, -40,
                                                   (t * self.pos).tolist())
            # overlay: line to the CURRENT goal (green = charger, orange = customer, red = restaurant)
            n = self.n_cur
            if self.seeking:
                color, mode, k = [0.1, 0.9, 0.3], "GO CHARGE", min(self.deliveries + 1, n)
            elif self.carrying:
                color, mode, k = [1.0, 0.6, 0.0], "DELIVERING", self.deliveries + 1
            else:
                color, mode, k = [0.95, 0.2, 0.2], "RETURNING TO RESTAURANT", max(1, self.deliveries)
            self._dbg_line = self.bc.addUserDebugLine(
                self.pos.tolist(), self.route_wps[self.route_ptr].tolist(), lineColorRGB=color, lineWidth=2.0,
                replaceItemUniqueId=self._dbg_line)
            label = (f"ORDER {k} of {n} | {mode} | battery {self.battery:.0f}%"
                     + (" | DELIVERED" if self.drop_steps > 0 else "")
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
        """Disconnect from PyBullet (the try/except keeps it safe to call more than once)."""
        try:
            self.bc.disconnect()
        except Exception:
            pass


# ------------------------------------------------------------ quick self-test----------------
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