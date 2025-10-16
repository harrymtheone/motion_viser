import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch


def play_motion(motion_path: str):
    data = np.load(motion_path)


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the TorchScript exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, model_path, policy_path):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt

        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.policy = torch.jit.load(policy_path)

        # Initialize renderer for depth images
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self._init_variables()

    def _init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt

        self.dof_pos = np.zeros(self.cfg.robot.num_joints)
        self.dof_vel = np.zeros(self.cfg.robot.num_joints)
        self.actions = np.zeros(self.cfg.sim.num_action)

        self.vae_hidden_states = torch.zeros(1, 1, 128)

        self.default_dof_pos = np.array(
            [0, 0.785, 0, -1.3, 0, -1, 0, 1.3, 0, 1, 0, -0.2, 0, 0, 0.4, -0.25, 0, -0.2, 0, 0, 0.4, -0.25, 0]
        )
        self.episode_length_buf = 0

        self.dof_activated = np.array(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        )

        # Initial command vel
        self.command_clock = np.array([0.0, 0.0])
        self.command_vel = np.array([0.0, 0.0, 0.0])

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        phase = (self.episode_length_buf * self.dt / 0.7) % 1.0
        self.command_clock[0] = np.sin(2 * np.pi * (phase + 0.))
        self.command_clock[1] = np.sin(2 * np.pi * (phase + 0.5))

        self.dof_pos[:] = self.data.sensordata[:23]
        self.dof_vel[:] = self.data.sensordata[23:46]

        obs = np.zeros((50,), dtype=np.float32)

        # Angular vel
        obs[0:3] = self.data.sensor("angular-velocity").data

        # Projected gravity
        obs[3:6] = self.quat_rotate_inverse(
            self.data.sensor("orientation").data, np.array([0, 0, -1])
        )

        # Command clock
        obs[6:8] = self.command_clock

        # Command velocity (scaled to match training normalization)
        # commands_scale = [obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel] = [2.0, 2.0, 1.0]
        obs[8:11] = self.command_vel * np.array([2.0, 2.0, 1.0])

        # Dof pos
        obs[11:24] = (self.dof_pos - self.default_dof_pos)[self.dof_activated] * 1.0

        # Dof vel
        obs[24:37] = self.dof_vel[self.dof_activated] * 0.05

        # Action (use RAW unclipped action to match training's last_action_output)
        obs[37:50] = self.actions

        # Get and display RGB image
        rgb = self.get_rgb_image()
        cv2.imshow("camera head", rgb)
        cv2.waitKey(1)

        return np.clip(obs, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def position_control(self):
        actions = np.clip(self.actions * self.cfg.sim.action_scale, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

        target_pos = self.default_dof_pos.copy()
        target_pos[self.dof_activated] += actions
        self.data.ctrl[:] = target_pos

    def torque_control(self):
        actions = np.clip(self.actions * self.cfg.sim.action_scale, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

        target_pos = self.default_dof_pos.copy()
        target_pos[self.dof_activated] += actions

        kp = np.array(
            [30, 30, 300, 200, 200, 100, 300, 200, 200, 100, 100, 55, 55, 30, 100, 30, 30, 55, 55, 30, 100, 30, 30]
        )

        kd = np.array(
            [1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 0.3, 0.3, 3, 3, 4, 5, 0.3, 0.3]
        )

        torques = kp * (target_pos - self.data.qpos[7:])
        torques[:] -= kd * self.data.qvel[6:]
        self.data.ctrl = torques

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        # self.setup_keyboard_listener()
        # self.listener.start()

        while self.data.time < self.cfg.sim.sim_duration and self.viewer.is_running():
            proprio = self.get_obs()

            self.actions[:], self.vae_hidden_states[:] = self.policy(
                torch.tensor(proprio, dtype=torch.float32).unsqueeze(0),
                self.vae_hidden_states
            )

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                self.position_control()
                # self.torque_control()

                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()

                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.episode_length_buf += 1

        self.viewer.close()

    def get_depth_image(self, camera_name: str = "head_camera") -> np.ndarray:
        """
        Capture depth image from specified camera.

        Args:
            camera_name (str): Name of the camera to capture from.

        Returns:
            np.ndarray: Depth image array (height x width).
        """
        self.renderer.update_scene(self.data, camera=camera_name)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        return depth

    def get_rgb_image(self, camera_name: str = "head_camera") -> np.ndarray:
        """
        Capture RGB image from specified camera.

        Args:
            camera_name (str): Name of the camera to capture from.

        Returns:
            np.ndarray: RGB image array (height x width x 3).
        """
        self.renderer.update_scene(self.data, camera=camera_name)
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render()
        return rgb

    @staticmethod
    def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion in (w, x, y, z).
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[0]
        q_vec = q[1:4]
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c


if __name__ == "__main__":
    play_motion('T1/data/walk2_subject3.npz')

    with torch.inference_mode():
        sim_cfg = SimToSimCfg()

        runner = MujocoRunner(
            cfg=sim_cfg,
            model_path="T1/robot/T1_serial.xml",
            policy_path="models/t1_dream_006_1800_jit.pt",
        )
        runner.run()
