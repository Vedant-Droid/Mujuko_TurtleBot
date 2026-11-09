#!/usr/bin/env python3
"""
Laptop Coordinate Controller Bridge (Wi-Fi TCP Client + 3D MuJoCo Arena)
========================================================================
Spawns your custom robot at the origin (0, 0) inside the MuJoCo arena and bridges
coordinate commands (like `4,4`) to the Raspberry Pi 4B (or a local embedded server).

Usage:
  1. Run directly on Laptop (auto-starts local sim server & opens 3D Arena):
       python3 laptop_controller.py

  2. Connect to physical Raspberry Pi 4B over Wi-Fi (while also watching in 3D Arena):
       python3 laptop_controller.py --ip <RASPBERRY_PI_IP>

Interactive Commands:
  - Type coordinates like `4,4` or `4 4` -> Robot moves to (4.0, 4.0) m from origin (0,0)
  - Type `origin` or `reset`             -> Resets robot back to origin (0, 0)
  - Type `stop`                          -> Immediately stops the motors
  - Type `q` or `quit`                   -> Disconnects and exits
"""

import argparse
import json
import math
import re
import socket
import sys
import threading
import time

BRIDGE_ARENA = 5.0  # ±5.0 m arena (10m x 10m floor) so coordinates like (4, 4) fit inside


def make_bridge_arena_xml() -> str:
    """
    Builds the MuJoCo XML scene with a 10m x 10m arena (±5.0 m walls),
    an origin marker at (0, 0), the green goal marker, and your custom robot
    spawned at the origin (0, 0, 0.049).
    """
    walls = f"""
      <geom name="wall_bottom" type="box" pos="0 {-BRIDGE_ARENA} 0.15"
            size="{BRIDGE_ARENA} 0.05 0.15" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_top" type="box" pos="0 {BRIDGE_ARENA} 0.15"
            size="{BRIDGE_ARENA} 0.05 0.15" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_left" type="box" pos="{-BRIDGE_ARENA} 0 0.15"
            size="0.05 {BRIDGE_ARENA} 0.15" rgba="0.5 0.5 0.5 1"/>
      <geom name="wall_right" type="box" pos="{BRIDGE_ARENA} 0 0.15"
            size="0.05 {BRIDGE_ARENA} 0.15" rgba="0.5 0.5 0.5 1"/>
    """
    return f"""
    <mujoco model="custom_robot_bridge_arena">

      <compiler angle="radian" meshdir="models"/>

      <option timestep="0.02" gravity="0 0 -9.81"/>

      <visual>
        <global offwidth="1200" offheight="900"/>
      </visual>

      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.82 0.84 0.86" rgb2="0.72 0.75 0.78"/>
        <material name="grid_mat" texture="grid" texrepeat="10 10" texuniform="true"
                  reflectance="0.05"/>

        <!-- 270 mm diameter base plate -->
        <mesh name="base" file="Base.stl" scale="0.001 0.001 0.001"/>
        <!-- 83 mm diameter wheels (scaled from 98.05 mm raw STL -> 0.0008465 in X/Z) -->
        <mesh name="left_wheel" file="TireL.stl" scale="0.0008465 0.001 0.0008465"/>
        <mesh name="right_wheel" file="TireR.stl" scale="0.0008465 0.001 0.0008465"/>
      </asset>

      <worldbody>

        <light pos="0 0 8" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>

        <geom name="floor" type="plane" size="{BRIDGE_ARENA + 1.0} {BRIDGE_ARENA + 1.0} 0.1"
              material="grid_mat"/>

        {walls}

        <!-- Origin (0, 0) Floor Marker -->
        <geom name="origin_marker" type="cylinder" pos="0 0 0.002" size="0.16 0.002"
              rgba="0.2 0.5 0.9 0.55" contype="0" conaffinity="0"/>

        <!-- Custom Robot spawned at Floor Origin (0, 0, 0.0415) for 83mm wheels -->
        <body name="robot" mocap="true" pos="0 0 0.0415">

          <geom name="base_bottom_visual"
                type="mesh"
                mesh="base"
                pos="0 0 0"
                euler="0 0 1.5707963"
                rgba="0.2 0.22 0.25 1"
                contype="0"
                conaffinity="0"/>

          <geom name="base_top_visual"
                type="mesh"
                mesh="base"
                pos="0 0 0.045"
                euler="0 0 1.5707963"
                rgba="0.2 0.22 0.25 1"
                contype="0"
                conaffinity="0"/>

          <body name="left_wheel" pos="0 0.110 0">
            <joint name="left_wheel_joint" type="hinge" axis="0 1 0"/>
            <geom name="left_wheel_visual"
                  type="mesh"
                  mesh="left_wheel"
                  pos="0.00477 -0.00143 0.01759"
                  euler="0 0 0"
                  rgba="0.15 0.15 0.15 1"
                  contype="0"
                  conaffinity="0"/>
          </body>

          <body name="right_wheel" pos="0 -0.110 0">
            <joint name="right_wheel_joint" type="hinge" axis="0 1 0"/>
            <geom name="right_wheel_visual"
                  type="mesh"
                  mesh="right_wheel"
                  pos="0.00477 -0.00143 -0.01759"
                  euler="0 0 0"
                  rgba="0.15 0.15 0.15 1"
                  contype="0"
                  conaffinity="0"/>
          </body>

          <!-- Front caster ball (+X forward, flush with 83mm wheels) -->
          <geom name="front_caster"
                type="sphere"
                pos="0.115 0 -0.0290"
                size="0.0125"
                rgba="0.75 0.15 0.15 1"
                contype="0"
                conaffinity="0"/>

          <!-- Rear caster ball (-X rear, flush with 83mm wheels) -->
          <geom name="rear_caster"
                type="sphere"
                pos="-0.115 0 -0.0290"
                size="0.0125"
                rgba="0.75 0.15 0.15 1"
                contype="0"
                conaffinity="0"/>

        </body>

        <!-- Target Goal Marker -->
        <body name="goal" mocap="true" pos="0 0 0.025">
          <geom type="cylinder" size="0.14 0.025"
                rgba="0.1 0.85 0.25 0.85" contype="0" conaffinity="0"/>
        </body>

      </worldbody>
    </mujoco>
    """


class PiBridgeClient:
    """TCP client on the laptop that sends coordinate commands to the Raspberry Pi 4B."""

    def __init__(self, ip: str, port: int = 5000, use_viewer: bool = True):
        self.ip = ip
        self.port = port
        self.sock = None
        self.file = None
        self.use_viewer = use_viewer
        self.model = None
        self.data = None
        self.viewer = None
        self.viewer_lock = threading.Lock()
        self.running = True

        if self.use_viewer:
            self._init_mujoco_twin()

    def _init_mujoco_twin(self):
        """Launches the 3D MuJoCo arena with the robot spawned immediately at (0, 0)."""
        try:
            import mujoco
            import mujoco.viewer

            self.mujoco = mujoco
            self.model = mujoco.MjModel.from_xml_string(make_bridge_arena_xml())
            self.data = mujoco.MjData(self.model)
            self.robot_mid = self.model.body("robot").mocapid[0]
            self.goal_mid = self.model.body("goal").mocapid[0]

            # Spawn robot at origin (0, 0, 0.0415) facing +X (yaw = 0)
            self.data.mocap_pos[self.robot_mid] = [0.0, 0.0, 0.0415]
            self.data.mocap_quat[self.robot_mid] = [1.0, 0.0, 0.0, 0.0]
            self.data.mocap_pos[self.goal_mid] = [0.0, 0.0, 0.025]
            mujoco.mj_forward(self.model, self.data)

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # Point camera at the arena center so robot at (0,0) and goals up to (4,4) are visible
            self.viewer.cam.lookat[:] = [1.0, 1.0, 0.0]
            self.viewer.cam.distance = 8.5
            self.viewer.cam.elevation = -45.0
            self.viewer.cam.azimuth = 135.0
            self.viewer.sync()

            # Keep viewer window responsive while waiting for terminal input
            threading.Thread(target=self._viewer_keepalive, daemon=True).start()
            print("[ARENA] MuJoCo 3D Arena launched — Robot spawned at origin (0.0, 0.0).")
        except Exception as exc:
            print(f"[ARENA] Could not launch MuJoCo viewer ({exc}). Continuing in terminal mode.")
            self.use_viewer = False

    def _viewer_keepalive(self):
        while self.running and self.viewer is not None and self.viewer.is_running():
            with self.viewer_lock:
                self.viewer.sync()
            time.sleep(1.0 / 30.0)

    def _update_twin(self, telem: dict):
        if not self.use_viewer or self.viewer is None:
            return
        with self.viewer_lock:
            x, y, yaw = telem["x"], telem["y"], telem["yaw_rad"]
            gx, gy = telem["goal_x"], telem["goal_y"]
            self.data.mocap_pos[self.robot_mid] = [x, y, 0.0415]
            self.data.mocap_quat[self.robot_mid] = [
                math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)
            ]
            self.data.qpos[0] = telem.get("left_wheel_rad", 0.0)
            self.data.qpos[1] = telem.get("right_wheel_rad", 0.0)
            self.data.mocap_pos[self.goal_mid] = [gx, gy, 0.025]
            self.mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()

    def connect(self):
        print(f"Connecting to Robot Server at {self.ip}:{self.port} ...")
        try:
            self.sock = socket.create_connection((self.ip, self.port), timeout=3.0)
        except (ConnectionRefusedError, TimeoutError, OSError):
            if self.ip in ("127.0.0.1", "localhost"):
                print(
                    "[AUTO-SIM] `pi_robot_server.py` was not running locally — "
                    "starting embedded simulation server on 127.0.0.1..."
                )
                from pi_robot_server import DiffDriveRobotServer

                self._local_server = DiffDriveRobotServer(
                    host="127.0.0.1", port=self.port, simulate=True
                )
                threading.Thread(target=self._local_server.serve_forever, daemon=True).start()
                time.sleep(0.25)
                self.sock = socket.create_connection((self.ip, self.port), timeout=3.0)
            else:
                raise

        self.sock.settimeout(2.0)
        self.file = self.sock.makefile("r", encoding="utf-8")
        telem = self.send_command({"cmd": "status"})
        self._update_twin(telem)
        print(
            f"Connected! Robot Spawned at Origin: "
            f"x={telem['x']:.2f} m, y={telem['y']:.2f} m, yaw={telem['yaw_deg']:.1f}°\n"
        )

    def send_command(self, payload: dict, timeout: float = 2.0) -> dict:
        self.sock.settimeout(timeout)
        raw = json.dumps(payload) + "\n"
        self.sock.sendall(raw.encode("utf-8"))
        line = self.file.readline()
        if not line:
            raise ConnectionError("Server closed the connection.")
        return json.loads(line)

    def run_preflight_check(self) -> bool:
        """
        Triggers LiDAR check and Left/Right wheel motor+encoder calibration on the Pi
        (and animates the wheel spin check in the MuJoCo 3D viewer) before entering coordinates.
        """
        print("---------------------------------------------------------")
        print(" Running Pre-Flight Check: LiDAR + Wheel Motor Calibration")
        print("---------------------------------------------------------")

        # Visually spin left then right wheel in the 3D viewer during calibration
        if self.use_viewer and self.viewer is not None:
            for angle in (0.4, 0.8, 0.4, 0.0):
                with self.viewer_lock:
                    self.data.qpos[0] = angle
                    self.mujoco.mj_forward(self.model, self.data)
                    self.viewer.sync()
                time.sleep(0.05)
            for angle in (0.4, 0.8, 0.4, 0.0):
                with self.viewer_lock:
                    self.data.qpos[1] = angle
                    self.mujoco.mj_forward(self.model, self.data)
                    self.viewer.sync()
                time.sleep(0.05)

        res = self.send_command({"cmd": "preflight"}, timeout=15.0)
        self._update_twin(res)
        pf = res.get("preflight", {})
        lidar = pf.get("lidar", {})
        motors = pf.get("motors", {})

        lidar_ok = lidar.get("lidar_ok", False)
        motors_ok = motors.get("motors_ok", False)

        print(
            f"  [LiDAR]  Status: {'OK' if lidar_ok else 'FAIL'} ({lidar.get('mode', 'N/A')}) | "
            f"Front={lidar.get('front_m', 0):.2f}m, Left={lidar.get('left_m', 0):.2f}m, "
            f"Rear={lidar.get('rear_m', 0):.2f}m, Right={lidar.get('right_m', 0):.2f}m"
        )
        print(
            f"  [Motors] Status: {'OK' if motors_ok else 'FAIL'} | "
            f"Left={motors.get('left_ticks', 0)} ticks (trim={motors.get('left_trim', 1.0):.3f}) | "
            f"Right={motors.get('right_ticks', 0)} ticks (trim={motors.get('right_trim', 1.0):.3f})"
        )
        print("---------------------------------------------------------\n")
        return bool(lidar_ok and motors_ok)

    def track_until_reached(self):
        """Polls live telemetry from the Pi while the robot drives to the target coordinate."""
        print("Press Ctrl+C at any time to emergency-stop the robot.")
        try:
            while True:
                telem = self.send_command({"cmd": "status"})
                self._update_twin(telem)

                sys.stdout.write(
                    f"\r  Pose: ({telem['x']:+6.2f}, {telem['y']:+6.2f}) m | "
                    f"Yaw: {telem['yaw_deg']:+6.1f}° | "
                    f"Target: ({telem['goal_x']:+5.2f}, {telem['goal_y']:+5.2f}) m | "
                    f"Dist: {telem['distance']:5.2f} m | "
                    f"State: {telem['status']:<19}"
                )
                sys.stdout.flush()

                if not telem["active"]:
                    print(f"\n[DONE] Robot status: {telem['status']}\n")
                    break

                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Sending stop command to robot...")
            self.send_command({"cmd": "stop"})

    def close(self):
        self.running = False
        if self.viewer is not None:
            self.viewer.close()
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass


def parse_coordinate_input(user_input: str):
    """Parse '4,4', '4, 4', or '4 4' into (float, float)."""
    parts = [p for p in re.split(r"[,\s]+", user_input.strip()) if p]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Laptop-to-Raspberry-Pi Coordinate Controller")
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="Raspberry Pi 4B Wi-Fi IP address (e.g., 192.168.1.50). Default: 127.0.0.1",
    )
    parser.add_argument("--port", type=int, default=5000, help="TCP port (default: 5000)")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Disable the 3D MuJoCo arena window (terminal-only mode)",
    )
    args = parser.parse_args()

    client = PiBridgeClient(ip=args.ip, port=args.port, use_viewer=not args.no_viewer)
    try:
        client.connect()
        client.run_preflight_check()

        print("Commands:")
        print("  x,y       -> Move robot to coordinate (x, y) in meters (e.g. 4,4 or 2,3)")
        print("  calibrate -> Re-run wheel motor calibration & LiDAR check")
        print("  origin    -> Reset current floor position as (0, 0)")
        print("  stop      -> Emergency stop")
        print("  q         -> Quit\n")

        while True:
            user_in = input("Enter target coordinate (x,y) > ").strip()
            if not user_in:
                continue

            cmd_lower = user_in.lower()
            if cmd_lower in ("q", "quit", "exit"):
                client.send_command({"cmd": "stop"})
                break
            elif cmd_lower in ("calibrate", "check", "lidar"):
                client.run_preflight_check()
                continue
            elif cmd_lower in ("origin", "reset"):
                telem = client.send_command({"cmd": "reset_origin"})
                client._update_twin(telem)
                print("Origin reset to (0.00, 0.00) at current floor position.\n")
                continue
            elif cmd_lower == "stop":
                client.send_command({"cmd": "stop"})
                print("Motors stopped.\n")
                continue

            coords = parse_coordinate_input(user_in)
            if coords is None:
                print("Invalid input. Enter coordinates like: 4,4 or 1.5, -2.0\n")
                continue

            gx, gy = coords
            client.send_command({"cmd": "goto", "x": gx, "y": gy})
            client.track_until_reached()

    except (ConnectionError, OSError) as exc:
        print(f"\n[ERROR] Connection failed: {exc}")
        print("Make sure `pi_robot_server.py` is running on the Raspberry Pi and on the same Wi-Fi.")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        client.close()


if __name__ == "__main__":
    main()
