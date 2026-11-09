#!/usr/bin/env python3
"""
Raspberry Pi 4B Robot Bridge Server (Wi-Fi TCP + Wheel Encoders)
=================================================================
Run this script on the Raspberry Pi 4B once the robot is placed on the floor.
The initial placement point on the floor is treated as the origin (0.0, 0.0, yaw=0.0).

Features:
- Reads Left & Right quadrature motor encoders to compute exact real-world odometry (x, y, yaw)
  and wheel rotation angles.
- Streams real-time encoder odometry back to the laptop so the MuJoCo 3D simulation
  mirrors the physical robot's exact movement on the floor.
- Closed-loop Go-To-Goal differential-drive controller calibrated for your custom robot:
    * Wheel diameter : 83 mm  (WHEEL_R = 0.0415 m, circumference = 0.2608 m)
    * Wheel track    : 220 mm (WHEEL_TRACK = 0.220 m)
"""

import argparse
import json
import math
import socket
import threading
import time

# =====================================================================
# ROBOT PHYSICAL CONSTANTS (from your Fusion 360 model)
# =====================================================================
WHEEL_R = 0.0415         # 83 mm tire diameter -> 0.0415 m radius
WHEEL_TRACK = 0.220      # 220 mm center-to-center distance between wheels
MAX_LIN_VEL = 0.25       # Max linear speed (m/s) at 100% PWM
MAX_ANG_VEL = 1.50       # Max angular speed (rad/s)

# Controller gains & thresholds
KP_LIN = 0.60            # Proportional gain for forward speed
KP_ANG = 1.80            # Proportional gain for steering angle
GOAL_TOLERANCE = 0.05    # Stop when within 5 cm of target
TURN_IN_PLACE_RAD = 0.35 # (~20 deg) Rotate in place first if heading error is large
CONTROL_HZ = 20.0        # 20 Hz control loop (dt = 0.05 s)

# =====================================================================
# RASPBERRY PI 4B GPIO PINOUT (BCM numbering)
# =====================================================================
# Left Motor Driver (BTS7960)
LEFT_RPWM_PIN = 12       # Forward PWM
LEFT_LPWM_PIN = 13       # Reverse PWM
LEFT_EN_PIN = 6          # Enable pin

# Right Motor Driver (BTS7960)
RIGHT_RPWM_PIN = 18      # Forward PWM
RIGHT_LPWM_PIN = 19      # Reverse PWM
RIGHT_EN_PIN = 26        # Enable pin

# Quadrature Wheel Encoders (Hall-effect A/B phases on motor shaft)
USE_ENCODERS = True
TICKS_PER_REV = 600      # Encoder ticks per 1 full wheel revolution (adjust to your motor's gear ratio * PPR)
LEFT_ENC_A = 17          # Left encoder phase A (BCM 17)
LEFT_ENC_B = 27          # Left encoder phase B (BCM 27)
RIGHT_ENC_A = 22         # Right encoder phase A (BCM 22)
RIGHT_ENC_B = 23         # Right encoder phase B (BCM 23)


def wrap_angle(angle: float) -> float:
    """Wrap angle in radians to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class RobotHardware:
    """
    Hardware interface for dual BTS7960 motor drivers + quadrature wheel encoders
    on Raspberry Pi 4B.
    """

    def __init__(self, simulate: bool = False):
        self.simulate = simulate
        self.gpio_available = False
        self.encoders_available = False
        self._last_left_ticks = 0
        self._last_right_ticks = 0

        # Calibration trim & polarity (updated by calibrate_and_check.py)
        self.left_trim = 1.0
        self.right_trim = 1.0
        self.left_encoder_sign = 1
        self.right_encoder_sign = 1
        self.load_calibration()

        if not simulate:
            try:
                from gpiozero import DigitalOutputDevice, PWMOutputDevice, RotaryEncoder

                # Initialize Motor PWM pins
                self.left_rpwm = PWMOutputDevice(LEFT_RPWM_PIN, frequency=1000)
                self.left_lpwm = PWMOutputDevice(LEFT_LPWM_PIN, frequency=1000)
                self.left_en = DigitalOutputDevice(LEFT_EN_PIN, initial_value=True)

                self.right_rpwm = PWMOutputDevice(RIGHT_RPWM_PIN, frequency=1000)
                self.right_lpwm = PWMOutputDevice(RIGHT_LPWM_PIN, frequency=1000)
                self.right_en = DigitalOutputDevice(RIGHT_EN_PIN, initial_value=True)
                self.gpio_available = True
                print("[HW] Raspberry Pi GPIO initialized (BTS7960 PWM mode).")

                # Initialize Quadrature Wheel Encoders
                if USE_ENCODERS:
                    self.left_enc = RotaryEncoder(LEFT_ENC_A, LEFT_ENC_B, max_steps=0)
                    self.right_enc = RotaryEncoder(RIGHT_ENC_A, RIGHT_ENC_B, max_steps=0)
                    self.encoders_available = True
                    print(
                        f"[HW] Wheel Encoders initialized on BCM ({LEFT_ENC_A},{LEFT_ENC_B}) & "
                        f"({RIGHT_ENC_A},{RIGHT_ENC_B}) | {TICKS_PER_REV} ticks/rev."
                    )
            except Exception as exc:
                print(f"[HW] GPIO/Encoder init failed ({exc}). Falling back to simulation mode.")
                self.simulate = True
        else:
            print("[HW] Running in simulation mode (synthetic encoder ticks).")

    def load_calibration(self):
        import os
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
        if os.path.exists(calib_path):
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    c = json.load(f)
                self.left_trim = float(c.get("left_trim", 1.0))
                self.right_trim = float(c.get("right_trim", 1.0))
                self.left_encoder_sign = int(c.get("left_encoder_sign", 1))
                self.right_encoder_sign = int(c.get("right_encoder_sign", 1))
            except Exception:
                pass

    def reset_encoders(self):
        if self.encoders_available and not self.simulate:
            self.left_enc.steps = 0
            self.right_enc.steps = 0
        self._last_left_ticks = 0
        self._last_right_ticks = 0

    def read_wheel_deltas(self, v_left_cmd: float, v_right_cmd: float, dt: float):
        """
        Returns (ds_left, ds_right, dphi_left, dphi_right):
          - ds_left, ds_right: linear distance (meters) traveled by each wheel over dt
          - dphi_left, dphi_right: wheel rotation angle (radians) over dt
        Uses real hardware encoder ticks when on the Pi, or kinematic velocity in --sim mode.
        """
        if self.encoders_available and not self.simulate:
            left_ticks = self.left_enc.steps
            right_ticks = self.right_enc.steps

            d_ticks_l = (left_ticks - self._last_left_ticks) * self.left_encoder_sign
            d_ticks_r = (right_ticks - self._last_right_ticks) * self.right_encoder_sign

            self._last_left_ticks = left_ticks
            self._last_right_ticks = right_ticks

            dphi_left = (2.0 * math.pi * d_ticks_l) / TICKS_PER_REV
            dphi_right = (2.0 * math.pi * d_ticks_r) / TICKS_PER_REV

            ds_left = dphi_left * WHEEL_R
            ds_right = dphi_right * WHEEL_R
            return ds_left, ds_right, dphi_left, dphi_right
        else:
            # Simulation fallback
            ds_left = v_left_cmd * dt
            ds_right = v_right_cmd * dt
            dphi_left = ds_left / WHEEL_R
            dphi_right = ds_right / WHEEL_R
            return ds_left, ds_right, dphi_left, dphi_right

    def set_wheel_velocities(self, v_left: float, v_right: float):
        """Convert desired wheel linear velocities (m/s) to trimmed PWM duty cycles [-1.0, 1.0]."""
        duty_left = max(-1.0, min(1.0, (v_left * self.left_trim) / MAX_LIN_VEL))
        duty_right = max(-1.0, min(1.0, (v_right * self.right_trim) / MAX_LIN_VEL))

        if self.gpio_available and not self.simulate:
            if duty_left >= 0:
                self.left_lpwm.value = 0.0
                self.left_rpwm.value = duty_left
            else:
                self.left_rpwm.value = 0.0
                self.left_lpwm.value = -duty_left

            if duty_right >= 0:
                self.right_lpwm.value = 0.0
                self.right_rpwm.value = duty_right
            else:
                self.right_rpwm.value = 0.0
                self.right_lpwm.value = -duty_right

    def stop(self):
        self.set_wheel_velocities(0.0, 0.0)


class DiffDriveRobotServer:
    """
    Maintains robot pose (x, y, yaw) from wheel encoders starting at floor origin (0, 0, 0),
    executes closed-loop Go-To-Goal control, and streams pose to laptop over Wi-Fi.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000, simulate: bool = False):
        self.host = host
        self.port = port
        self.hw = RobotHardware(simulate=simulate)

        # Pose state in meters & radians (Origin = where robot is placed on floor)
        self.lock = threading.Lock()
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.left_wheel_rad = 0.0
        self.right_wheel_rad = 0.0

        # Goal state
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.active_goal = False
        self.status = "IDLE_AT_ORIGIN"

        # Current velocities
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.v_left_cmd = 0.0
        self.v_right_cmd = 0.0
        self.running = True

    def reset_origin(self):
        """Sets the robot's current floor position as (0, 0, yaw=0) and zeroes encoders."""
        with self.lock:
            self.hw.stop()
            self.hw.reset_encoders()
            self.x = 0.0
            self.y = 0.0
            self.yaw = 0.0
            self.left_wheel_rad = 0.0
            self.right_wheel_rad = 0.0
            self.goal_x = 0.0
            self.goal_y = 0.0
            self.active_goal = False
            self.v_cmd = 0.0
            self.w_cmd = 0.0
            self.v_left_cmd = 0.0
            self.v_right_cmd = 0.0
            self.status = "ORIGIN_RESET"
        print("[POSE] Floor origin & encoders reset to (0.00, 0.00, 0.0 deg)")

    def set_goal(self, gx: float, gy: float):
        """Assigns a new (x, y) target coordinate in meters."""
        with self.lock:
            self.goal_x = float(gx)
            self.goal_y = float(gy)
            self.active_goal = True
            self.status = "MOVING_TO_GOAL"
        print(f"[GOAL] Navigating to ({gx:.2f}, {gy:.2f}) m")

    def stop_robot(self):
        with self.lock:
            self.active_goal = False
            self.v_cmd = 0.0
            self.w_cmd = 0.0
            self.v_left_cmd = 0.0
            self.v_right_cmd = 0.0
            self.status = "STOPPED"
            self.hw.stop()
        print("[STOP] Motors stopped.")

    def get_telemetry(self) -> dict:
        with self.lock:
            dist = math.hypot(self.goal_x - self.x, self.goal_y - self.y)
            return {
                "x": round(self.x, 3),
                "y": round(self.y, 3),
                "yaw_rad": round(self.yaw, 4),
                "yaw_deg": round(math.degrees(self.yaw), 1),
                "left_wheel_rad": round(self.left_wheel_rad, 3),
                "right_wheel_rad": round(self.right_wheel_rad, 3),
                "goal_x": round(self.goal_x, 3),
                "goal_y": round(self.goal_y, 3),
                "distance": round(dist, 3),
                "v": round(self.v_cmd, 3),
                "w": round(self.w_cmd, 3),
                "active": self.active_goal,
                "status": self.status,
            }

    def _control_step(self, dt: float):
        with self.lock:
            # 1. Read encoder wheel travel since last step and update odometry (x, y, yaw)
            ds_l, ds_r, dphi_l, dphi_r = self.hw.read_wheel_deltas(
                self.v_left_cmd, self.v_right_cmd, dt
            )
            self.left_wheel_rad += dphi_l
            self.right_wheel_rad += dphi_r

            ds = 0.5 * (ds_r + ds_l)
            dyaw = (ds_r - ds_l) / WHEEL_TRACK
            mid_yaw = self.yaw + 0.5 * dyaw
            self.x += ds * math.cos(mid_yaw)
            self.y += ds * math.sin(mid_yaw)
            self.yaw = wrap_angle(self.yaw + dyaw)

            # 2. If no active goal, keep motors stopped
            if not self.active_goal:
                self.v_left_cmd = 0.0
                self.v_right_cmd = 0.0
                self.hw.stop()
                return

            # 3. Compute distance & heading error to target coordinate
            dx = self.goal_x - self.x
            dy = self.goal_y - self.y
            dist = math.hypot(dx, dy)

            if dist <= GOAL_TOLERANCE:
                self.active_goal = False
                self.v_cmd = 0.0
                self.w_cmd = 0.0
                self.v_left_cmd = 0.0
                self.v_right_cmd = 0.0
                self.status = "GOAL_REACHED"
                self.hw.stop()
                print(
                    f"[REACHED] Arrived at ({self.x:.2f}, {self.y:.2f}) m "
                    f"(Target: {self.goal_x:.2f}, {self.goal_y:.2f})"
                )
                return

            target_heading = math.atan2(dy, dx)
            heading_error = wrap_angle(target_heading - self.yaw)

            # Phase 1: Rotate in place if facing away from goal
            if abs(heading_error) > TURN_IN_PLACE_RAD:
                v = 0.0
                w = KP_ANG * heading_error
                self.status = "ROTATING_TO_HEADING"
            else:
                # Phase 2: Drive forward & steer toward goal
                v = KP_LIN * dist * math.cos(heading_error)
                w = KP_ANG * heading_error
                self.status = "DRIVING_TO_GOAL"

            v = max(-MAX_LIN_VEL, min(MAX_LIN_VEL, v))
            w = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, w))

            if abs(v) > 1e-3:
                v = math.copysign(max(abs(v), 0.06), v)

            self.v_cmd = v
            self.w_cmd = w

            # Differential-drive inverse kinematics -> left & right wheel linear speeds
            self.v_left_cmd = v - w * (WHEEL_TRACK / 2.0)
            self.v_right_cmd = v + w * (WHEEL_TRACK / 2.0)
            self.hw.set_wheel_velocities(self.v_left_cmd, self.v_right_cmd)

    def _control_loop(self):
        period = 1.0 / CONTROL_HZ
        last_time = time.monotonic()
        while self.running:
            now = time.monotonic()
            dt = now - last_time
            last_time = now
            self._control_step(dt)
            elapsed = time.monotonic() - now
            time.sleep(max(0.0, period - elapsed))

    def _handle_client(self, conn: socket.socket, addr):
        print(f"[WIFI] Laptop connected from {addr[0]}:{addr[1]}")
        buffer = ""
        try:
            with conn:
                while self.running:
                    data = conn.recv(1024)
                    if not data:
                        break
                    buffer += data.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        response = self._process_command(line)
                        conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except ConnectionResetError:
            pass
        finally:
            print(f"[WIFI] Laptop {addr[0]} disconnected. Stopping motors for safety.")
            self.stop_robot()

    def run_preflight(self) -> dict:
        """Runs LiDAR diagnostic and Left/Right wheel motor+encoder calibration."""
        from calibrate_and_check import LidarChecker, run_motor_and_encoder_calibration

        with self.lock:
            self.active_goal = False
            self.hw.stop()
            lidar_checker = LidarChecker(simulate=self.hw.simulate)
            lidar_rep = lidar_checker.run_diagnostic()
            motor_rep = run_motor_and_encoder_calibration(self.hw)
            self.hw.load_calibration()
            self.x = 0.0
            self.y = 0.0
            self.yaw = 0.0
            self.left_wheel_rad = 0.0
            self.right_wheel_rad = 0.0
            self.status = "CALIBRATED_READY"
        return {"lidar": lidar_rep, "motors": motor_rep}

    def _process_command(self, raw_line: str) -> dict:
        try:
            msg = json.loads(raw_line)
        except json.JSONDecodeError:
            return {"ok": False, "error": "Invalid JSON"}

        cmd = msg.get("cmd", "").lower()
        preflight_data = None
        if cmd == "goto":
            self.set_goal(float(msg["x"]), float(msg["y"]))
        elif cmd == "reset_origin":
            self.reset_origin()
        elif cmd == "stop":
            self.stop_robot()
        elif cmd in ("preflight", "calibrate"):
            preflight_data = self.run_preflight()
        elif cmd == "status":
            pass
        else:
            return {"ok": False, "error": f"Unknown cmd '{cmd}'"}

        telemetry = self.get_telemetry()
        if preflight_data is not None:
            telemetry["preflight"] = preflight_data
        telemetry["ok"] = True
        return telemetry

    def serve_forever(self):
        ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        ctrl_thread.start()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(2)
            print("=========================================================")
            print(f" Raspberry Pi 4B Robot Server listening on {self.host}:{self.port}")
            print(" Initial floor placement set as origin: (x=0.0, y=0.0)")
            print(" Waiting for coordinates from laptop_controller.py ...")
            print("=========================================================")

            try:
                while self.running:
                    conn, addr = server.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client, args=(conn, addr), daemon=True
                    )
                    client_thread.start()
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Stopping motors and closing server.")
            finally:
                self.running = False
                self.hw.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi 4B Diff-Drive Bridge Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind IP address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="TCP port (default: 5000)")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run in simulation mode (test on laptop without Raspberry Pi GPIO)",
    )
    args = parser.parse_args()

    robot_server = DiffDriveRobotServer(host=args.host, port=args.port, simulate=args.sim)
    robot_server.serve_forever()
