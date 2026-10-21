import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

ARENA = 2.2
LIDAR_MAX = 3.0
ROBOT_R = 0.15
OBSTACLES = [
    (-0.55, -0.15, 0.23),
    (0.55, 0.55, 0.25),
    (-0.65, 0.95, 0.22),
    (0.95, -0.75, 0.25),
    (0.15, -1.0, 0.22),
]

def make_xml():
    walls = f"""
      <geom name="wall_bottom" type="box" pos="0 {-ARENA} 0.15"
            size="{ARENA} 0.05 0.15" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_top" type="box" pos="0 {ARENA} 0.15"
            size="{ARENA} 0.05 0.15" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_left" type="box" pos="{-ARENA} 0 0.15"
            size="0.05 {ARENA} 0.15" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_right" type="box" pos="{ARENA} 0 0.15"
            size="0.05 {ARENA} 0.15" rgba="0.5 0.5 0.5 1"/>
    """
    obstacles = "".join(
        f'<geom name="obstacle{i}" type="cylinder" '
        f'pos="{x} {y} 0.12" size="{r} 0.12" '
        f'rgba="0.8 0.25 0.15 1"/>'
        for i, (x, y, r) in enumerate(OBSTACLES)
    )
    return f"""
    <mujoco model="turtlebot_navigation">
      <option timestep="0.02" gravity="0 0 -9.81"/>
      <visual>
        <global offwidth="1000" offheight="800"/>
      </visual>
      <worldbody>
        <light pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="5 5 0.1"
              rgba="0.85 0.85 0.85 1"/>
        {walls}
        {obstacles}
        <body name="robot" mocap="true" pos="-1.6 -1.6 0.08">
          <geom type="cylinder" size="0.15 0.08"
                rgba="0.1 0.35 0.9 1"/>
          <geom type="box" pos="0.12 0 0.02" size="0.07 0.035 0.025"
                rgba="1 0.8 0.1 1"/>
        </body>
        <body name="goal" mocap="true" pos="1.4 1.4 0.04">
          <geom type="cylinder" size="0.13 0.025"
                rgba="0.1 0.8 0.2 1"/>
        </body>
      </worldbody>
    </mujoco>
    """

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

class TurtleBotNavEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, max_steps=300):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.n_lidar = 16
        self.dt = 0.1
        self.model = mujoco.MjModel.from_xml_string(make_xml())
        self.data = mujoco.MjData(self.model)
        self.robot_mid = self.model.body("robot").mocapid[0]
        self.goal_mid = self.model.body("goal").mocapid[0]
        self.viewer = None

        # Lidar values [0,1], distance [0,1], angle [-1,1],
        # and previous action [-1,1].
        low = np.array([0.0] * 17 + [-1.0] * 3, dtype=np.float32)
        high = np.ones(20, dtype=np.float32)
        high[17:] = 1.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.x = self.y = self.yaw = 0.0
        self.goal = np.zeros(2)
        self.prev_dist = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def _distance(self):
        return float(np.linalg.norm(self.goal - [self.x, self.y]))

    def _ray_distance(self, angle):
        ux, uy = math.cos(angle), math.sin(angle)
        distances = [LIDAR_MAX]

        # Distance to the four arena boundaries.
        if ux > 1e-8:
            distances.append((ARENA - self.x) / ux)
        elif ux < -1e-8:
            distances.append((-ARENA - self.x) / ux)
        if uy > 1e-8:
            distances.append((ARENA - self.y) / uy)
        elif uy < -1e-8:
            distances.append((-ARENA - self.y) / uy)

        # Ray-circle intersections with obstacles.
        for ox, oy, radius in OBSTACLES:
            dx, dy = ox - self.x, oy - self.y
            projection = dx * ux + dy * uy
            discriminant = projection**2 - (dx**2 + dy**2 - radius**2)
            if discriminant >= 0:
                t = projection - math.sqrt(discriminant)
                if t >= 0:
                    distances.append(t)

        return float(np.clip(min(d for d in distances if d >= 0),
                             0.0, LIDAR_MAX))

    def _observe(self):
        lidar = np.array([
            self._ray_distance(
                self.yaw + 2 * math.pi * i / self.n_lidar
            ) / LIDAR_MAX
            for i in range(self.n_lidar)
        ], dtype=np.float32)

        dx, dy = self.goal[0] - self.x, self.goal[1] - self.y
        dist = math.hypot(dx, dy)
        angle = wrap_angle(math.atan2(dy, dx) - self.yaw)

        return np.concatenate([
            lidar,
            [min(dist / 4.5, 1.0), angle / math.pi],
            self.last_action,
        ]).astype(np.float32)

    def _sync_model(self):
        self.data.mocap_pos[self.robot_mid] = [self.x, self.y, 0.08]
        self.data.mocap_quat[self.robot_mid] = [
            math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)
        ]
        self.data.mocap_pos[self.goal_mid] = [
            self.goal[0], self.goal[1], 0.04
        ]
        mujoco.mj_forward(self.model, self.data)

    def _collides(self):
        if (abs(self.x) > ARENA - ROBOT_R or
                abs(self.y) > ARENA - ROBOT_R):
            return True
        for ox, oy, radius in OBSTACLES:
            if math.hypot(self.x - ox, self.y - oy) < ROBOT_R + radius:
                return True
        return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.x, self.y, self.yaw = -1.6, -1.6, 0.0

        # Sample a goal that is not inside an obstacle.
        for _ in range(1000):
            gx, gy = self.np_random.uniform(-1.75, 1.75, size=2)
            if (math.hypot(gx - self.x, gy - self.y) > 0.8 and
                all(math.hypot(gx - ox, gy - oy) > r + 0.25
                    for ox, oy, r in OBSTACLES)):
                self.goal = np.array([gx, gy], dtype=float)
                break

        self.steps = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.prev_dist = self._distance()
        self._sync_model()
        if self.render_mode == "human":
            self.render()
        return self._observe(), {"goal": self.goal.copy()}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        v = float(action[0]) * 0.22
        w = float(action[1]) * 1.5

        # Differential-drive kinematics, with a 0.1 s control step.
        self.x += v * math.cos(self.yaw) * self.dt
        self.y += v * math.sin(self.yaw) * self.dt
        self.yaw = wrap_angle(self.yaw + w * self.dt)
        self.last_action = action.astype(np.float32)
        self.steps += 1

        dist = self._distance()
        
        ##Collision
        collision = self._collides()
        success = dist < 0.20

        reward = 3.0 * (self.prev_dist - dist) - 0.01
        if collision:
            reward -= 10.0
        if success:
            reward += 10.0
        self.prev_dist = dist

        self._sync_model()
        if self.render_mode == "human":
            self.render()

        terminated = collision or success
        truncated = self.steps >= self.max_steps
        return self._observe(), reward, terminated, truncated, {
            "success": success, "collision": collision, "distance": dist
        }

    def render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        time.sleep(1 / 30)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
