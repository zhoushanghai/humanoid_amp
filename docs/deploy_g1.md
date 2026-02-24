# 从零实现基于 G1 的 AMP 训练系统架构指南

本文档旨在为在一个全新的强化学习项目（而非直接在 MimicKit 内部）中实现针对 Unitree G1 机器人的对抗运动先验 (AMP，Adversarial Motion Priors) 控制策略提供全面的架构设计和实现指南。

---

## 核心系统架构图谱

实现一个完整的 AMP 系统，你需要在新项目中重点开发和对接以下四大核心模块：

1.  **物理仿真层 (Environment layer)**
2.  **参考运动数据层 (Motion Library layer)**
3.  **强化学习算法引擎 (RL Algorithm layer - PPO+Discriminator)**
4.  **状态与奖励计算通道 (State & Reward builders)**

---

## 第一模块：物理仿真层 (Environment Layer)

无论你选择 Isaac Gym 还是 Isaac Sim 作为底层物理引擎，你的 `G1AMPEnv` 类必须处理好以下细节：



### 1.2 `Reset` 逻辑 (生命周期) 的特殊处理
在 AMP 算法中，传统的“每次都从直立姿态开始”会导致训练效率极低。
*   **Reference State Initialization (RSI) - 参考状态初始化**：
    每次环境复位时，你必须从“运动数据层 (Motion Library)”中**随机抽取任何一个时间点 $t$**，将机器人直接强行放置到该时间的姿势和速度状态上，并添加微小的均匀分布随机噪声。这让网络能快速学习到动作序列的每一个阶段。

### 1.3 `Done` 逻辑 (终止条件)
*   超过预设的最大步数 (例如 1000 步或 10 秒)。
*   **提前终止 (Early Termination)**：当机器人的“非脚部接触体”（如躯干、头部、膝盖、手臂）发生地面接触碰撞时，判定为摔倒，立刻给出 Done 信号终止。这在防止学出爬行动作时非常关键。

---

## 第二模块：参考运动数据层 (Motion Library)

AMP 是以数据为驱动的，你需要一个专门的类来管理用于训练的“参考教材”(Expert Data)。

### 2.1 数据格式与加载
*   准备针对 G1 骨架进行过重定向 (Retargeting) 的人类动捕数据文件（如 `.npy` 或 `.npz`）。
*   每帧数据必须包含：`root_pos`, `root_rot` (通常是四元数), 以及所有 23 个活动 `joint_positions`。

### 2.2 运动学插值采样器 (Kinematic Sampler)
强化学习是以极高频率运行的（如 50Hz 或 100Hz），而动捕数据通常是 30fps，两者不匹配的，所以需要写一个插值系统：
*   给定一个运动片段 ID 和指定绝对时间 $t$。
*   对于位置使用线性插值 (Linear Interpolation)。
*   对于根节点和关节的旋转值，必须使用**球面线性插值 (Slerp)**，否则强行相加会导致四元数崩溃。
*   **导数计算**：如果数据集中没有包含速度，采样器需要通过差分计算出当前时间的 `root_vel`, `root_ang_vel` 和 `joint_vel`。

---

## 第三模块：状态构建与奖励机制 (State & Reward)

这是连接“环境”和“神经网络”的核心数据流配置，必须极其严谨。

### 3.1 策略网观测空间 (Actor Observation Space)
这是 G1 为了完成“不要摔倒/受控”所需的信息：
*   **自身状态**：
    *   根节点距地面高度 (`root_height`)。
    *   根节点**局部**倾斜旋转矩阵或四元数 (`root_rot`)——**切记必须剥离开全局的偏航角 (Yaw)** 和全局 X/Y 坐标，以保证观测平移/旋转不变性。
    *   **局部坐标系**下的根节点线速度和角速度。
    *   所有自由度的角度 (`dof_pos`) 和角速度 (`dof_vel`)。
    *   四肢末端（手、脚）相对于根节点的位置坐标。

### 3.2 判别器状态空间 (Discriminator State/Observation)
判别器负责区分网络学出来的动作是否有“人味”。这需要不同的数据馈送策略：
*   **引入时序性：历史缓冲池 (History Buffer)**：
    判别器**绝对不能**只读取当前帧的姿态，因为静止的跌倒姿势和跳跃准备姿势可能是一模一样的。你必须维护一个队列大小为 $N$ (通常 $N \ge 5$) 的循环缓冲区。
*   你喂给判别器的巨大向量需包含连续过去的 $N$ 个时间步的：`局部关节位姿`, `局部关节速度`, `局部根节点速度`。
*   **坐标系一致性**：丢给判别器的无论是真数据（从 Motion Library 取出），还是假数据（当前仿真器的姿势），其所有数据都必须转换到**当前帧根节点朝向的局部坐标系下**。

### 3.3 奖励设计模型 (Reward Structure)
总奖励 $=$ AMP对抗奖励 $+$ 生存奖励 $+$ 动作惩罚。
*   **AMP 鉴别器回报**：
    $$r_{amp} = - \ln(\max(1 - D(s_{hist}), 10^{-5})) $$ （或是其他变体，其中 $D$ 是判别器判为真/专家的概率）。
*   **动作惩罚/正则化 (Action Regularization)**：
    非常关键！如果不惩罚动作变化率 (`action_rate_penalty`) 或整体输出力矩 (`torque_penalty`)，G1 极有可能学出高频颤抖的伪合法人形动作，这种策略放到真机上会瞬间烧毁电机。

---

## 第四模块：RL 算法引擎层 (PPO + Discriminator)

在你的算法包里需要写一个魔改版本的 PPO。

### 4.1 独立的三大网络构建
1.  **Actor (策略网络)**：`Input[Actor_Obs] -> MLP(e.g., [1024, 512]) -> Output[Joint_Targets]`
2.  **Critic (价值网络)**：`Input[Actor_Obs] -> MLP(e.g., [1024, 512]) -> Output[Value(1D)]`
3.  **Discriminator (判别网络)**：`Input[Disc_Hist_State] -> MLP(e.g., [1024, 512]) -> Output[Logit(1D)]`（判为“真实参考数据”的概率对数值）。

### 4.2 数据收集与训练闭环 (Training Loop)
*   **Step 1: 收集 Rollout 经验**
    利用 Actor 在环境中跑，存下 `(obs, act, reward, done)`。注意这里的 `reward` 是根据 Discriminator 实时推断的风格得分计算出来的。
*   **Step 2: 收集真实样本**
    从 Motion Library 的真实动捕数据集中，随机切出与 Rollout 等量的“真实历史状态切片 (Expert States)”。
*   **Step 3: 更新 Discriminator (判别器)**
    使用二分类交叉熵损失：
    *   目标是让拿到的 Expert States 预测概率逼近 1。
    *   让缓冲区拿到的 Rollout Agent States 预测概率逼近 0。
    *   **重要踩坑点**：务必添加 **梯度截断惩罚 (Gradient Penalty)** 以防止判别器早期过快收敛导致梯度消失（Actor 没法得到有效指导奖励）。
*   **Step 4: 更新 Actor & Critic (PPO)**
    走标准的 PPO 剪裁优势更新流程。

---

## 结语

将这四大模块在新代码库中解耦并分层实现，你的 G1 控制策略架构就立住了。
**首要难点**在于数据的精确同步：尤其是保证提供给判别器的局部坐标状态，无论是来自物理引擎在线算出的，还是从pkl/npz提取的真数据，都必须在参考平移、欧拉角展开方式、和旋转推导数学公式上**达到丝毫不差的像素级对齐**，否则系统将完全崩溃。
