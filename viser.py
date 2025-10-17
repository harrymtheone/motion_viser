import time

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF


class MotionPlayer:

    def __init__(self, server: viser.ViserServer, urdf_path: str, motion_data_path: str):
        """
        Initialize the MotionPlayer.
        
        Args:
            server: Viser server instance
            urdf_path: Path to the robot URDF file
            motion_data_path: Path to the motion default (.npz file)
        """
        self.server = server
        self.urdf_path = urdf_path

        # Load motion default
        self._load_motion_data(motion_data_path)

        # Load robot model
        self._load_robot()

        # Setup GUI controls
        self._setup_gui()

        # Playback state
        self.is_playing = False
        self.current_frame = 0
        self.playback_speed = 1.0
        self.last_update_time = time.time()

        # Initialize robot to first frame
        self.update_robot_pose(0)

        print(f"MotionPlayer initialized successfully!")
        print(f"Motion: {self.num_frames} frames at {self.fps} Hz ({self.duration:.2f}s)")
        print(f"Server: http://localhost:8080")

    def _load_motion_data(self, motion_data_path: str) -> None:
        """Load motion default from npz file."""
        data = np.load(motion_data_path, allow_pickle=True)

        # Extract position default
        qpos = data["qpos"]
        self.root_pos = qpos[:, :3]  # Root position (x, y, z)
        self.root_quat = qpos[:, 3:7]  # Root quaternion (w, x, y, z)
        self.joint_pos = qpos[:, 7:]  # Joint positions

        # Extract velocity default
        qvel = data["qvel"]
        self.root_lin_vel = qvel[:, :3]  # Root linear velocity
        self.root_ang_vel = qvel[:, 3:6]  # Root angular velocity
        self.joint_vel = qvel[:, 6:]  # Joint velocities

        # Motion metadata
        self.fps = float(data['frequency'])
        self.num_frames = len(qpos)
        self.duration = self.num_frames / self.fps
        self.dt = 1.0 / self.fps

    def _load_robot(self) -> None:
        """Load robot URDF and create visualization."""
        # Add ground plane with checker texture
        self._add_ground_plane()
        
        # Create a parent frame for robot positioning
        self.robot_frame = self.server.scene.add_frame("/robot", show_axes=True)

        # Load URDF
        urdf = URDF.load(self.urdf_path)

        # Create URDF visualizer
        self.viser_urdf = ViserUrdf(self.server, urdf, root_node_name="/robot")
    
    def _add_ground_plane(self) -> None:
        """Add a textured ground plane similar to MuJoCo."""
        # Add a semi-transparent ground plane mesh
        ground_size = 50.0
        ground_mesh = trimesh.creation.box(
            extents=[ground_size, ground_size, 0.01]  # Very thin box (XY plane)
        )
        
        # Set the visual color for the mesh
        ground_mesh.visual.vertex_colors = [51, 77, 102, 204]  # RGBA (80% opacity)
        
        self.server.scene.add_mesh_trimesh(
            name="/ground/plane",
            mesh=ground_mesh,
            position=(0.0, 0.0, -0.02),  # Base plane lower to avoid z-fighting
        )
        
        # Add a grid on top to visualize the checker pattern
        self.server.scene.add_grid(
            name="/ground/grid",
            width=50.0,
            height=50.0,
            plane="xy",  # Horizontal plane (XY plane at z=0)
            cell_color=(40, 60, 80),  # Darker checker color (similar to rgb2)
            cell_thickness=2.0,
            cell_size=1.0,
        )

    def _setup_gui(self) -> None:
        """Setup GUI controls for motion playback."""
        # Playback controls in a folder
        with self.server.gui.add_folder("Playback Controls"):
            self.play_button = self.server.gui.add_button("▶ Play")
            self.pause_button = self.server.gui.add_button("⏸ Pause")

            self.speed_slider = self.server.gui.add_slider(
                "Speed",
                min=0.1,
                max=3.0,
                step=0.1,
                initial_value=1.0
            )
        
        # Frame slider at the bottom - larger and more prominent
        self.server.gui.add_markdown("---")  # Separator
        self.server.gui.add_markdown("**Timeline**")
        
        self.frame_slider = self.server.gui.add_slider(
            "Frame",
            min=0,
            max=self.num_frames - 1,
            step=1,
            initial_value=0,
            marks=((0, "Start"), (self.num_frames - 1, "End"))
        )

        # Connect callbacks
        self.play_button.on_click(lambda _: self.play())
        self.pause_button.on_click(lambda _: self.pause())
        self.speed_slider.on_update(lambda _: self._update_speed())
        self.frame_slider.on_update(lambda _: self._on_frame_slider_changed())

    def _on_frame_slider_changed(self) -> None:
        """Called when frame slider is manually adjusted."""
        self.current_frame = int(self.frame_slider.value)

    def _update_speed(self) -> None:
        """Update playback speed from slider."""
        self.playback_speed = self.speed_slider.value

    def update_robot_pose(self, frame_idx: int) -> None:
        """
        Update robot pose to match the specified frame.

        Args:
            frame_idx: Frame index in the motion default
        """
        # Ensure frame index is valid
        frame_idx = frame_idx % self.num_frames

        # Update robot base position and orientation
        position = self.root_pos[frame_idx]
        quaternion = self.root_quat[frame_idx]  # wxyz format

        self.robot_frame.position = tuple(position)
        self.robot_frame.wxyz = tuple(quaternion)

        # Update joint positions
        joint_config = self.joint_pos[frame_idx]
        self.viser_urdf.update_cfg(joint_config)

    def play(self) -> None:
        """Start motion playback."""
        self.is_playing = True
        print("Playing motion...")

    def pause(self) -> None:
        """Pause motion playback."""
        self.is_playing = False
        print("Motion paused")

    def update(self) -> None:
        """
        Update animation state. Should be called in the main loop.
        """
        current_time = time.time()
        elapsed = current_time - self.last_update_time

        if self.is_playing and elapsed >= (self.dt / self.playback_speed):
            # Advance to next frame
            self.current_frame = (self.current_frame + 1) % self.num_frames

            # Update robot pose
            self.update_robot_pose(self.current_frame)

            # Update frame slider
            self.frame_slider.value = self.current_frame

            self.last_update_time = current_time

        elif not self.is_playing:
            # If paused, still update pose if frame changed
            self.update_robot_pose(self.current_frame)

    def run(self) -> None:
        """
        Run the main animation loop.
        """
        print("Starting animation loop...")

        try:
            while True:
                self.update()

                # Small sleep to prevent CPU spinning
                if self.is_playing:
                    time.sleep(0.001)
                else:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nShutting down...")


def main():
    """Main entry point."""
    # Initialize Viser server
    server = viser.ViserServer()

    # Configuration
    urdf_path = "/home/harry/projects/amp_viser/T1/T1_serial.urdf"
    motion_data_path = "T1/data/walkturn.npz"

    # Create motion player
    player = MotionPlayer(server, urdf_path, motion_data_path)

    # Run animation loop
    player.run()


if __name__ == "__main__":
    main()
