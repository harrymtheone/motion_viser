import time

import mujoco
import mujoco.viewer
import numpy as np


class MujocoMotionPlayer:

    def __init__(self, model_path: str, motion_data_path: str):
        """
        Initialize the MuJoCo Motion Player.
        
        Args:
            model_path: Path to the MuJoCo model XML file
            motion_data_path: Path to the motion data (.npz file)
        """
        self.model_path = model_path

        # Load motion data
        self._load_motion_data(motion_data_path)

        # Load MuJoCo model
        self._load_model()

        # Playback state
        self.current_frame = 0
        self.playback_speed = 1.0
        self.last_update_time = time.time()

        print(f"MujocoMotionPlayer initialized successfully!")
        print(f"Motion: {self.num_frames} frames at {self.fps} Hz ({self.duration:.2f}s)")
        print(f"Robot has {self.model.nq} DOFs (qpos) and {self.model.nv} DOFs (qvel)")

    def _load_motion_data(self, motion_data_path: str) -> None:
        """Load motion data from npz file."""
        data = np.load(motion_data_path, allow_pickle=True)

        # Extract position data
        qpos = data["qpos"]
        self.root_pos = qpos[:, :3]  # Root position (x, y, z)
        self.root_quat = qpos[:, 3:7]  # Root quaternion (w, x, y, z)
        self.joint_pos = qpos[:, 7:]  # Joint positions

        # Extract velocity data
        qvel = data["qvel"]
        self.root_lin_vel = qvel[:, :3]  # Root linear velocity
        self.root_ang_vel = qvel[:, 3:6]  # Root angular velocity
        self.joint_vel = qvel[:, 6:]  # Joint velocities

        # Store full qpos and qvel for convenience
        self.qpos_data = qpos
        self.qvel_data = qvel

        # Motion metadata
        self.fps = float(data['frequency'])
        self.num_frames = len(qpos)
        self.duration = self.num_frames / self.fps
        self.dt = 1.0 / self.fps

        print(f"Loaded motion data: qpos shape={qpos.shape}, qvel shape={qvel.shape}")

    def _load_model(self) -> None:
        """Load MuJoCo model."""
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Set initial pose to first frame
        self._set_robot_state(0)

    def _set_robot_state(self, frame_idx: int) -> None:
        """
        Set robot state to match the specified frame.
        
        Args:
            frame_idx: Frame index in the motion data
        """
        # Ensure frame index is valid
        frame_idx = frame_idx % self.num_frames

        # Update qpos and qvel
        qpos = self.qpos_data[frame_idx]
        qvel = self.qvel_data[frame_idx]

        # Verify dimensions match
        if len(qpos) != self.model.nq:
            print(f"Warning: qpos size mismatch. Data has {len(qpos)}, model expects {self.model.nq}")
            # Copy what we can
            copy_len = min(len(qpos), self.model.nq)
            self.data.qpos[:copy_len] = qpos[:copy_len]
        else:
            self.data.qpos[:] = qpos

        if len(qvel) != self.model.nv:
            print(f"Warning: qvel size mismatch. Data has {len(qvel)}, model expects {self.model.nv}")
            # Copy what we can
            copy_len = min(len(qvel), self.model.nv)
            self.data.qvel[:copy_len] = qvel[:copy_len]
        else:
            self.data.qvel[:] = qvel

        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)

    def update(self) -> None:
        """
        Update animation state. Advances to next frame based on timing.
        """
        current_time = time.time()
        elapsed = current_time - self.last_update_time

        if elapsed >= (self.dt / self.playback_speed):
            # Advance to next frame
            self.current_frame = (self.current_frame + 1) % self.num_frames

            # Update robot state
            self._set_robot_state(self.current_frame)

            self.last_update_time = current_time

            # Print progress every second
            if self.current_frame % int(self.fps) == 0:
                print(f"Frame {self.current_frame}/{self.num_frames} ({self.current_frame / self.fps:.1f}s)")

    def run(self) -> None:
        """
        Run the motion playback with MuJoCo viewer.
        """
        print("Starting motion playback...")

        # Launch the viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Reset timer
            self.last_update_time = time.time()

            # Run simulation loop
            while viewer.is_running():
                # Update to next frame
                self.update()

                # Sync the viewer
                viewer.sync()

                # Small sleep to prevent CPU spinning
                time.sleep(0.001)

        print("Playback stopped")


def main():
    """Main entry point."""
    # Create motion player
    player = MujocoMotionPlayer(
        model_path="T1/robot/T1_serial.xml",
        motion_data_path="T1/data/walkturn.npz",
    )

    # Run playback
    player.run()


if __name__ == "__main__":
    main()
