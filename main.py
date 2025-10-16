"""
MuJoCo Motion Player with Keyboard Controls

Simple motion player without GUI dependencies.

Controls:
    SPACE       : Play/Pause
    ← →         : Previous/Next frame (pause)
    [ ]         : Speed down/up (0.5x/2x)
    , .         : Jump 1 second backward/forward (no pause)
    ESC         : Exit
"""

import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np


class MuJoCoMotionPlayer:
    """MuJoCo Motion Player with keyboard controls."""

    def __init__(self, model_path: str, motion_data_path: str):
        """
        Initialize the MuJoCo Motion Player.
        
        Args:
            model_path: Path to the MuJoCo model XML file
            motion_data_path: Path to the motion data (.npz file)
        """
        # Load motion data
        self._load_motion_data(motion_data_path)

        # Load MuJoCo model
        self._load_model(model_path)

        # Playback state
        self.current_frame = 0
        self.playback_speed = 1.0
        self.last_update_time = time.time()
        self.is_playing = True
        self.loop_motion = True

        print(f"✓ Motion Player initialized!")
        print(f"  Frames: {self.num_frames} at {self.fps} Hz ({self.duration:.2f}s)")
        print(f"  DOFs: {self.model.nq} (qpos), {self.model.nv} (qvel)")

    def _load_motion_data(self, motion_data_path: str) -> None:
        """Load motion data from npz file."""
        data = np.load(motion_data_path, allow_pickle=True)

        self.qpos_data = data["qpos"]
        self.qvel_data = data["qvel"]

        self.fps = float(data['frequency'])
        self.num_frames = len(self.qpos_data)
        self.duration = self.num_frames / self.fps
        self.dt = 1.0 / self.fps

        print(f"  Loaded: qpos{self.qpos_data.shape}, qvel{self.qvel_data.shape}")

    def _load_model(self, model_path: str) -> None:
        """Load MuJoCo model."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._set_robot_state(0)

    def _set_robot_state(self, frame_idx: int) -> None:
        """Set robot state to a specific frame."""
        frame_idx = frame_idx % self.num_frames

        qpos = self.qpos_data[frame_idx]
        qvel = self.qvel_data[frame_idx]

        # Copy data (handle size mismatches gracefully)
        copy_len = min(len(qpos), self.model.nq)
        self.data.qpos[:copy_len] = qpos[:copy_len]

        copy_len = min(len(qvel), self.model.nv)
        self.data.qvel[:copy_len] = qvel[:copy_len]

        mujoco.mj_forward(self.model, self.data)

    def update(self) -> None:
        """Update animation - advance frames if playing."""
        if not self.is_playing:
            return

        current_time = time.time()
        elapsed = current_time - self.last_update_time

        if elapsed >= (self.dt / self.playback_speed):
            # Advance frame
            self.current_frame += 1

            # Handle looping
            if self.current_frame >= self.num_frames:
                if self.loop_motion:
                    self.current_frame = 0
                else:
                    self.current_frame = self.num_frames - 1
                    self.is_playing = False  # Stop at end

            self._set_robot_state(self.current_frame)
            self.last_update_time = current_time
            
            # Print progress
            time_sec = self.current_frame / self.fps
            progress_pct = (self.current_frame / self.num_frames) * 100
            status = "▶" if self.is_playing else "⏸"
            print(f"{status} Frame {self.current_frame + 1}/{self.num_frames} | "
                  f"{time_sec:.2f}s/{self.duration:.2f}s | "
                  f"{progress_pct:.1f}% | "
                  f"Speed: {self.playback_speed:.2f}x")

    def toggle_play_pause(self) -> None:
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        self.last_update_time = time.time()
        status = "▶ Playing" if self.is_playing else "⏸ Paused"
        current_time = self.current_frame / self.fps
        print(f"{status} at {current_time:.2f}s (frame {self.current_frame + 1}/{self.num_frames})")

    def jump_frames(self, num_frames: int, pause: bool) -> None:
        """Jump forward/backward by delta frames."""
        self.current_frame = (self.current_frame + num_frames) % self.num_frames
        self._set_robot_state(self.current_frame)
        self.last_update_time = time.time()
        if pause:
            self.is_playing = False
        print(f"⏩ Frame {self.current_frame + 1}/{self.num_frames}")

    def change_speed(self, factor: float) -> None:
        """Change playback speed by multiplying by factor."""
        self.playback_speed *= factor
        self.playback_speed = max(0.1, min(10.0, self.playback_speed))  # Clamp
        print(f"⏩ Speed: {self.playback_speed:.2f}x")

    def reset(self) -> None:
        """Reset to beginning."""
        self.current_frame = 0
        self._set_robot_state(0)
        self.last_update_time = time.time()
        print("⏮ Reset to start")

    def handle_keyboard(self, keycode: int) -> None:
        """Handle keyboard input."""

        # Map keycodes to actions
        key_map = {
            glfw.KEY_SPACE: lambda: self.toggle_play_pause(),
            glfw.KEY_LEFT: lambda: self.jump_frames(num_frames=-1, pause=True),
            glfw.KEY_RIGHT: lambda: self.jump_frames(num_frames=1, pause=True),
            glfw.KEY_LEFT_BRACKET: lambda: self.change_speed(0.5),
            glfw.KEY_RIGHT_BRACKET: lambda: self.change_speed(2.0),
            glfw.KEY_COMMA: lambda: self.jump_frames(num_frames=int(-self.fps), pause=False),
            glfw.KEY_PERIOD: lambda: self.jump_frames(num_frames=int(self.fps), pause=False),
        }

        if keycode in key_map:
            action = key_map[keycode]
            if action:
                action()

    def run(self) -> None:
        """Run the motion player with interactive viewer."""
        print("\n" + "=" * 60)
        print("  MUJOCO MOTION PLAYER")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE       : Play/Pause")
        print("  ← →         : Previous/Next frame (pause)")
        print("  [ ]         : Speed down/up (0.5x/2x)")
        print("  , .         : Jump 1 second backward/forward (no pause)")
        print("  ESC         : Exit")
        print("=" * 60)
        print()

        # Launch viewer with keyboard callback
        viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.handle_keyboard
        )

        # Set camera
        with viewer.lock():
            viewer.cam.azimuth = 90
            viewer.cam.distance = 3.5
            viewer.cam.elevation = -15

        self.last_update_time = time.time()

        # Main loop
        try:
            while viewer.is_running():
                # Update animation
                self.update()

                # Sync viewer
                viewer.sync()

                # Small sleep to prevent CPU spinning
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n⏹ Stopped by user")
        finally:
            viewer.close()

        print("✓ Playback finished\n")


def main():
    """Main entry point."""
    player = MuJoCoMotionPlayer(
        model_path="T1/robot/T1_serial.xml",
        motion_data_path="T1/data/walkturn.npz",
    )

    player.run()


if __name__ == "__main__":
    main()
