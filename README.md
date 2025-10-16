# Motion Visualizer

Motion visualization tools for humanoid robots using MuJoCo.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### MuJoCo Player with Keyboard Controls

```bash
python main.py
```

**Keyboard Controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Play/Pause |
| `←` `→` | Previous/Next frame (pause) |
| `[` `]` | Speed down/up (0.5x/2x) |
| `,` `.` | Jump 1 second backward/forward (no pause) |
| `ESC` | Exit |

### Configuration

Edit the paths in `main.py`:

```python
player = MuJoCoMotionPlayer(
    model_path="T1/robot/T1_serial.xml",
    motion_data_path="T1/data/walkturn.npz",
)
```

## Motion Data Format

Motion data is stored in `.npz` files with:
- `qpos`: Joint positions (N×DOF array)
- `qvel`: Joint velocities (N×DOF array)  
- `frequency`: Frame rate (Hz)

## Project Structure

```
motion_viser/
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── main.py               # MuJoCo player with keyboard controls
└── T1/                   # Robot model and data
    ├── robot/
    │   └── T1_serial.xml # MuJoCo robot model
    └── data/             # Motion capture data (.npz files)
```
