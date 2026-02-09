# 项目文档 - G1 AMP 训练

## 环境配置

```bash
# RTX 5090 需要手动指定 GPU 和 Vulkan
export CUDA_VISIBLE_DEVICES=0
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

---

## 数据转换

将 LAFAN1 CSV 转换为 NPZ 格式：

```bash
conda activate retarget
cd /home/hz/g1/humanoid_amp/motions
python data_convert.py
```

当前配置：
- 输入: `datasets/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv`
- 帧范围: 100-700 (600帧 @ 30fps = 20秒)
- URDF: `usd/g1_29dof_rev_1_0.urdf` (与仿真一致)
- 输出: `motions/G1_walk_lafan1.npz`

---

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Walk-Direct-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 20000
```

---

## 播放

```bash
CUDA_VISIBLE_DEVICES=0 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Walk-Direct-v0 \
  --checkpoint <checkpoint_path> \
  --num_envs 4 \
  --algorithm AMP
```

---

## 录制视频

```bash
CUDA_VISIBLE_DEVICES=0 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Walk-Direct-v0 \
  --checkpoint <checkpoint_path> \
  --num_envs 1 \
  --algorithm AMP \
  --video \
  --video_length 300
```

---

## 监控训练 (TensorBoard)

如果直接运行 `tensorboard` 报错，请使用：
```bash
python -m tensorboard.main --logdir logs/skrl/g1_amp_walk
```
