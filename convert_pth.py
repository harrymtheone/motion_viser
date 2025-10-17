import mujoco
import numpy as np
import torch


def inv_quat(quat):
    # type: (torch.Tensor) -> torch.Tensor
    scaling = torch.tensor([-1, -1, -1, 1], device=quat.device)
    return quat * scaling


def transform_by_quat(v, quat):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    v = v.reshape(-1, 3)
    quat = quat.reshape(-1, 4)
    qvec = quat[:, :3]
    t = qvec.cross(v, dim=-1) * 2
    return v + quat[:, 3:] * t + qvec.cross(t, dim=-1)


def cvt(path: str):
    data = np.load(path, allow_pickle=True)

    start_frame_idx = 100
    end_frame_idx = 265

    data_length = end_frame_idx - start_frame_idx
    fps = data['frequency'].item()
    qpos = torch.from_numpy(data['qpos'])[start_frame_idx: end_frame_idx]
    qvel = torch.from_numpy(data['qvel'])[start_frame_idx: end_frame_idx]

    root_pos_w = qpos[:, :3]
    root_quat_w = qpos[:, [4, 5, 6, 3]]  # wxyz -> xyzw
    root_lin_vel_w = qvel[:, :3]
    root_ang_vel_w = qvel[:, 3:6]

    joint_pos = qpos[:, 7:]
    joint_vel = qpos[:, 6:]

    root_lin_vel_b = transform_by_quat(root_lin_vel_w, inv_quat(root_quat_w))
    root_ang_vel_b = transform_by_quat(root_ang_vel_w, inv_quat(root_quat_w))

    base_height = root_pos_w[:, 2]
    projected_gravity = transform_by_quat(
        torch.tensor([0, 0, -1.]).repeat(data_length, 1),
        inv_quat(root_quat_w)
    )

    knee_pos_b = torch.zeros(data_length, 3)
    feet_pos_b = torch.zeros(data_length, 6)  # 3 for left foot + 3 for right foot

    mj_model = mujoco.MjModel.from_xml_path("T1/robot/T1_serial.xml")  # noqa
    mj_data = mujoco.MjData(mj_model)

    # Get body IDs for the feet
    left_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    right_foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")

    for frame_idx in range(data_length):
        mj_data.qpos[:] = qpos[frame_idx]
        mj_data.qvel[:] = qvel[frame_idx]

        mujoco.mj_forward(mj_model, mj_data)

        # Get feet positions in world frame
        left_foot_pos_w = torch.from_numpy(mj_data.xpos[left_foot_id]).float()
        right_foot_pos_w = torch.from_numpy(mj_data.xpos[right_foot_id]).float()

        # Transform feet positions to body frame (relative to root)
        left_foot_pos_b = left_foot_pos_w - root_pos_w[frame_idx]
        left_foot_pos_b = transform_by_quat(left_foot_pos_b, inv_quat(root_quat_w[frame_idx]))
        right_foot_pos_b = right_foot_pos_w - root_pos_w[frame_idx]
        right_foot_pos_b = transform_by_quat(right_foot_pos_b, inv_quat(root_quat_w[frame_idx]))

        # Store in buffer: [left_x, left_y, left_z, right_x, right_y, right_z]
        feet_pos_b[frame_idx, :3] = left_foot_pos_b
        feet_pos_b[frame_idx, 3:] = right_foot_pos_b

    torch.save(
        {
            "fps": fps,
            "weight": 1.,
            "root_pos_w": root_pos_w,
            "root_quat_w": root_quat_w,
            "root_lin_vel_w": root_lin_vel_w,
            "root_ang_vel_w": root_ang_vel_w,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "root_lin_vel_b": root_lin_vel_b,
            "root_ang_vel_b": root_ang_vel_b,
            "base_height": base_height,
            "projected_gravity": projected_gravity,
            "knee_pos_b": knee_pos_b,
            "feet_pos_b": feet_pos_b,
        },
        "../parkour_genesis/data/step_in_place/stepinplace1.pth",
    )


if __name__ == '__main__':
    cvt("T1/default/stepinplace1.npz")
