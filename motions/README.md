# Motion files

The motion files are in NumPy-file format that contains data from the skeleton DOFs and bodies that perform the motion.

The data (accessed by key) is described in the following table, where:

* `N` is the number of motion frames recorded
* `D` is the number of skeleton DOFs
* `B` is the number of skeleton bodies

| Key | Dtype | Shape | Description |
| --- | ---- | ----- | ----------- |
| `fps` | int64 | () | FPS at which motion was sampled |
| `dof_names` | unicode string | (D,) | Skeleton DOF names |
| `body_names` | unicode string | (B,) | Skeleton body names |
| `dof_positions` | float32 | (N, D) | Skeleton DOF positions |
| `dof_velocities` | float32 | (N, D) | Skeleton DOF velocities |
| `body_positions` | float32 | (N, B, 3) | Skeleton body positions |
| `body_rotations` | float32 | (N, B, 4) | Skeleton body rotations (as `wxyz` quaternion) |
| `body_linear_velocities` | float32 | (N, B, 3) | Skeleton body linear velocities |
| `body_angular_velocities` | float32 | (N, B, 3) | Skeleton body angular velocities |

## Motion visualization

The `motion_viewer.py` file allows to visualize the skeleton motion recorded in a motion file.

Open an terminal in the `motions` folder and run the following command.

```bash
python motion_viewer.py --file MOTION_FILE_NAME.npz
```

See `python motion_viewer.py --help` for available arguments.

G1 example:
```
dof_names:
  Data type: <U26
  Data shape: (29,)
  Joint names:
    1. left_hip_pitch_joint
    2. right_hip_pitch_joint
    3. waist_yaw_joint
    4. left_hip_roll_joint
    5. right_hip_roll_joint
    6. waist_roll_joint
    7. left_hip_yaw_joint
    8. right_hip_yaw_joint
    9. waist_pitch_joint
    10. left_knee_joint
    11. right_knee_joint
    12. left_shoulder_pitch_joint
    13. right_shoulder_pitch_joint
    14. left_ankle_pitch_joint
    15. right_ankle_pitch_joint
    16. left_shoulder_roll_joint
    17. right_shoulder_roll_joint
    18. left_ankle_roll_joint
    19. right_ankle_roll_joint
    20. left_shoulder_yaw_joint
    21. right_shoulder_yaw_joint
    22. left_elbow_joint
    23. right_elbow_joint
    24. left_wrist_roll_joint
    25. right_wrist_roll_joint
    26. left_wrist_pitch_joint
    27. right_wrist_pitch_joint
    28. left_wrist_yaw_joint
    29. right_wrist_yaw_joint

body_names:
  Data type: <U25
  Data shape: (39,)
  Body part names:
    1. pelvis
    2. imu_in_pelvis
    3. left_hip_pitch_link
    4. pelvis_contour_link
    5. right_hip_pitch_link
    6. waist_yaw_link
    7. left_hip_roll_link
    8. right_hip_roll_link
    9. waist_roll_link
    10. left_hip_yaw_link
    11. right_hip_yaw_link
    12. torso_link
    13. left_knee_link
    14. right_knee_link
    15. d435_link
    16. head_link
    17. imu_in_torso
    18. left_shoulder_pitch_link
    19. logo_link
    20. mid360_link
    21. right_shoulder_pitch_link
    22. left_ankle_pitch_link
    23. right_ankle_pitch_link
    24. left_shoulder_roll_link
    25. right_shoulder_roll_link
    26. left_ankle_roll_link
    27. right_ankle_roll_link
    28. left_shoulder_yaw_link
    29. right_shoulder_yaw_link
    30. left_elbow_link
    31. right_elbow_link
    32. left_wrist_roll_link
    33. right_wrist_roll_link
    34. left_wrist_pitch_link
    35. right_wrist_pitch_link
    36. left_wrist_yaw_link
    37. right_wrist_yaw_link
    38. left_rubber_hand
    39. right_rubber_hand
```
