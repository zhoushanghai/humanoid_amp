# humanoid_amp

单任务版本的 G1 AMP 仓库，只保留 `Isaac-G1-AMP-Poprioception-Direct-v0`。

## 安装

```bash
pip install -e .
```

## 训练

快速冒烟：

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Poprioception-Direct-v0 \
  --algorithm AMP \
  --num_envs 64 \
  --max_iterations 1 \
  --headless
```

完整训练：

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Poprioception-Direct-v0 \
  --algorithm AMP \
  --num_envs 4096 \
  --max_iterations 5000000 \
  --headless
```

## Play

```bash
python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Poprioception-Direct-v0 \
  --algorithm AMP \
  --checkpoint logs/skrl/g1_amp_poprioception/<run_name>/checkpoints/agent_last.pt \
  --num_envs 1 \
  --video \
  --video_length 300
```



cd ~/g1/humanoid_amp
conda activate g1_amp
python motions/data_convert.py \
  --csv ../../tool/GMR/dataset/01_01_poses.csv \
  --urdf g1_model/urdf/g1_29dof_rev_1_0.urdf \
  --meshes g1_model/urdf \
  --output motions/01_01_poses.npz


python motions/motion_viewer.py \
  --file motions/01_01_poses.npz