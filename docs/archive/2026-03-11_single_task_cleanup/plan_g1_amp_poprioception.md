# Plan: G1-AMP-Poprioception 基础探索任务

- Date: 2026-03-10
- Task Name: `G1-AMP-Poprioception`
- Planned Gym ID: `Isaac-G1-AMP-Poprioception-Direct-v0`
- Base Task: `Isaac-G1-AMP-Deploy-Direct-v0`
- Status: In Progress

## Objective

基于当前 `Isaac-G1-AMP-Deploy-Direct-v0`，做一个最小可训练的基础探索任务，用来先产出第一个稳定模型。

第一版只解决这三件事：

- 每个机器人拥有一个独立的 `3m x 3m` 房间。
- 每个房间随机生成 `1~3` 个固定障碍物，类型为立方体或圆柱体。
- 奖励采用 `style reward + exploration reward`，其中探索奖励由上肢有效接触、表面网格覆盖和几何权重构成。

## Confirmed Decisions

### 命名

- 任务名统一为 `G1-AMP-Poprioception`。
- Gym ID 统一为 `Isaac-G1-AMP-Poprioception-Direct-v0`。
- 保留 `Poprioception` 这个拼写，不主动改成 `Proprioception`。

### 实现隔离

- 除 `__init__.py` 中新增 task 注册项外，新任务全部使用独立文件实现。
- 不把新逻辑继续堆进现有 `g1_amp_env.py` 或 `g1_amp_env_cfg.py`。
- 旧任务文件只作为参考实现。

### 参考基线

- 任务注册参考 [__init__.py](/home/hz/g1/humanoid_amp/__init__.py#L85)。
- 环境配置参考 [g1_amp_env_cfg.py](/home/hz/g1/humanoid_amp/g1_amp_env_cfg.py#L176) 的 `G1AmpDeployEnvCfg`。
- 场景、观测、奖励入口参考 [g1_amp_env.py](/home/hz/g1/humanoid_amp/g1_amp_env.py#L158) [g1_amp_env.py](/home/hz/g1/humanoid_amp/g1_amp_env.py#L292) [g1_amp_env.py](/home/hz/g1/humanoid_amp/g1_amp_env.py#L376)。
- AMP agent 初始超参数参考 [agents/skrl_g1_deploy_amp_cfg.yaml](/home/hz/g1/humanoid_amp/agents/skrl_g1_deploy_amp_cfg.yaml#L1)。
- 机器人已启用 contact sensor，见 [g1_cfg.py](/home/hz/g1/humanoid_amp/g1_cfg.py#L10)。

## Scope

### In Scope

- 新增独立 task 注册项。
- 复用 Deploy 的 AMP 训练主干和 policy observation 结构。
- 每个环境创建独立房间和静态障碍物。
- 实现以下探索奖励：
  - 上肢有效接触点数量奖励
  - 新表面网格奖励
  - 几何权重奖励

### Out of Scope

- perception model
- 自由空间扫掠奖励
- 联合训练
- 下蹲动作专门支持
- 动态障碍物
- 任意 3D 倾斜摆放的障碍物姿态

## Environment Spec

### 房间

- 每个环境一个独立房间。
- 房间尺寸：`3.0m x 3.0m`
- 墙体高度：`1.5m`
- 墙体厚度：`0.08m`
- 机器人出生点：房间中心附近
- 地面继续复用当前环境中的 ground plane 生成逻辑

### 障碍物

- 每个环境随机生成 `1~3` 个障碍物。
- 类型：`cube` 或 `cylinder`
- 运动状态：固定不动
- 姿态：第一版只做绕 `z` 轴的随机 `yaw`

尺寸范围：

- 立方体边长：`0.40m ~ 0.80m`
- 圆柱半径：`0.20m ~ 0.35m`
- 立方体 / 圆柱高度：`0.50m ~ 1.00m`

布局约束：

- 与墙体保留安全距离：`0.25m`
- 与机器人出生点保留最小距离：`0.75m`
- 障碍物表面之间最小间距：`0.5m`

### 场景生成策略

- 不在运行时动态增删 prim。
- 每个环境预分配 `3` 个 obstacle slot。
- 每次 reset 时决定每个 slot 是否启用，以及其类型、位置、尺寸、yaw。

这样更适合当前 `clone_environments(copy_from_source=False)` 的工作方式，也更容易先做稳定基线。

当前实现说明：

- 由于 Isaac Lab 中单个刚体的 `xformOp:scale` 随机化只适合在 physics 启动前进行，第一版代码把障碍物尺寸改为 `startup` 时按环境采样一次。
- 每次 `reset` 仍然会重新采样：
  - 启用数量 `1~3`
  - slot 对应类型 `cube / cylinder`
  - 位置
  - `yaw`

## Training Defaults

第一版尽量少改动现有训练骨架：

- 继续使用 Deploy 的 AMP 训练结构。
- 继续使用当前 policy observation 思路，不重新设计新输入接口。
- 关闭 `rew_track_vel`
- 关闭 `rew_track_ang_vel_z`
- 关闭 command 相关输入
- 保留 AMP motion prior
- 保留 policy history stacking

也就是说，第一版不做速度命令跟踪，只做探索。

## Reward Design

### Style Reward

- 沿用当前 AMP 判别器奖励。
- agent 初始超参数先从 Deploy 配置复制。

### Exploration Reward

#### 1. 有效接触部位

只有以下机器人部位触碰障碍物时，才允许触发探索奖励：

- 手
- 前臂
- 上臂

实现默认按以下 body name 统计：

- `left_rubber_hand`
- `right_rubber_hand`
- `left_elbow_link`
- `right_elbow_link`
- `left_shoulder_yaw_link`
- `right_shoulder_yaw_link`

这样可以避免用腿、躯干“撞上去刷分”。

#### 2. Contact Count Reward

- 统计机器人与障碍物的有效接触点数量。
- 只统计来自上述上肢部位的接触。

#### 3. Novel Surface Grid Reward

- 将障碍物表面离散成网格。
- 每当首次触达未访问格子时，给予一次性奖励。
- 同一格子在同一 episode 内重复触碰不重复给分。
- 默认表面网格边长：`0.10m`

#### 4. Geometry-weighted Reward

新格子的奖励权重：

- 普通表面：`1`
- 棱边 / 圆柱边缘：`3`
- 尖角 / 拐点：`5`

第一版几何标签约定：

- 立方体：
  - 面内部：`1`
  - 靠边区域：`3`
  - 角点区域：`5`
- 圆柱体：
  - 侧面普通区域：`1`
  - 顶部/底部圆环边缘：`3`
  - 不额外定义 `5` 分角点

## Planned Files

### 注册

- `__init__.py`
  - 新增 `Isaac-G1-AMP-Poprioception-Direct-v0` 的 task 注册项

### 独立实现文件

- `g1_amp_poprioception_env_cfg.py`
  - 新环境配置类
  - 房间参数、障碍物参数、奖励权重、表面网格参数

- `g1_amp_poprioception_env.py`
  - 新环境主类
  - `_setup_scene()`、`_get_observations()`、`_get_rewards()`、reset 随机化

- `g1_amp_poprioception_scene.py`
  - 房间墙体创建
  - obstacle slot 创建
  - reset 时的布局采样

- `g1_amp_poprioception_rewards.py`
  - contact count reward
  - surface grid novelty reward
  - geometry weight 映射
  - 接触点到局部网格索引的辅助逻辑

- `g1_amp_poprioception_constants.py`
  - 房间尺寸默认值
  - 障碍物尺寸范围
  - grid cell size = `0.10m`
  - 几何权重 `1 / 3 / 5`
  - 允许计奖的 body name 列表

- `agents/skrl_g1_amp_poprioception_cfg.yaml`
  - 从 Deploy agent 配置复制出的独立配置
  - 只改实验目录和必要的 reward 权重

- `motions/motion_poprioception.yaml`
  - 新任务专用的 motion prior 配置文件
  - 第一版先独立于 `motions/motion_config.yaml` 维护

## Acceptance Criteria

- 环境能正常创建，不出现 room / obstacle prim 报错
- 每个环境都能看到独立的 `3m x 3m` 房间
- 每个环境 reset 后都有 `1~3` 个随机障碍物
- 障碍物不明显穿墙、穿地或彼此重叠
- 训练能正常启动，不出现 observation shape mismatch
- reward log 中至少能看到：
  - 接触点计数
  - 新格子数量
  - 加权探索奖励
- 机器人能学到基础接触行为，而不是只站立或直接摔倒

## Missing Info

下面这些信息仍建议在真正开始实现前补齐，否则实现过程中还会反复回头改：

- Contact Count Reward 的统计口径
  - 按接触点数直接计数
  - 还是按 body-object 接触对计数
  - 是否要做 per-step 上限
- 表面网格坐标化细节
  - 立方体如何从局部坐标映射到面网格
  - 圆柱体是否统计顶面、底面中心区域
- 几何边角区域的判定阈值
  - 靠边区域宽度
  - 角点区域半径
- 训练默认参数是否直接沿用 Deploy
  - `episode_length_s`
  - `num_actor_observations`
  - `task_reward_weight / style_reward_weight`
- runtime 中 `body_names` 是否与 URDF 推断完全一致

## Recommended Defaults

如果你不额外指定，建议第一版先按下面默认值实现：

- `episode_length_s = 20.0`
- `num_actor_observations = 5`
- `surface_grid_cell_size = 0.10`
- `task_reward_weight = 0.5`
- `style_reward_weight = 0.5`
- `motion_prior_file = motions/motion_poprioception.yaml`
- `contact_count_mode = body_object_pairs`
- `contact_count_per_step_cap = 4`
- 有效接触默认包含：
  - `left_rubber_hand`
  - `right_rubber_hand`
  - `left_elbow_link`
  - `right_elbow_link`
  - `left_shoulder_yaw_link`
  - `right_shoulder_yaw_link`

## Checklist

- ✅ ~~在 `__init__.py` 中新增 `Isaac-G1-AMP-Poprioception-Direct-v0`~~
- ✅ ~~新建 `g1_amp_poprioception_env_cfg.py`~~
- ✅ ~~新建 `g1_amp_poprioception_env.py`~~
- ✅ ~~新建 `g1_amp_poprioception_scene.py`~~
- ✅ ~~新建 `g1_amp_poprioception_rewards.py`~~
- ✅ ~~新建 `g1_amp_poprioception_constants.py`~~
- ✅ ~~新建 `agents/skrl_g1_amp_poprioception_cfg.yaml`~~
- ✅ ~~实现房间与 obstacle slot~~
- ✅ ~~实现 `1~3` 个障碍物的 reset 随机化~~
- ✅ ~~实现障碍物尺寸与 yaw 随机化~~
- ✅ ~~实现障碍物布局约束~~
- ✅ ~~实现仅上肢有效的接触过滤~~
- ✅ ~~实现表面网格离散与首次触达判定~~
- ✅ ~~实现 `1 / 3 / 5` 几何权重逻辑~~
- ✅ ~~在 `_get_rewards()` 中接入探索奖励日志~~
- ✅ ~~验证训练初始化和第一轮短训练~~

## Completion Summary

- 已完成最小训练冒烟验证：
  - `python -m humanoid_amp.train --task Isaac-G1-AMP-Poprioception-Direct-v0 --algorithm AMP --num_envs 4 --max_iterations 1 --headless`
- 已完成最小 play 验证：
  - 使用 `best_agent.pt` 成功回放并录制 `60` 帧视频
- 运行时修复：
  - 补齐每个环境下的 `Room` / `Obstacles` / `slot_*` 父级 prim，解决 regex spawn 时的缺失父 prim 报错
- 当前结论：
  - 新任务已经可以完成环境创建、短训练、checkpoint 加载和 play 录像闭环
