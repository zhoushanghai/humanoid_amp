# Plan: History Reset Warm-Start Fix

## 问题
Actor 历史 Buffer 重置时全部清零，导致策略在 Episode 开始时接收到异常的"零值历史"，
破坏 RunningStandardScaler 的统计数据，造成含历史输入时训练不收敛。

## 解决方案
使用延迟填充策略（Lazy Warm-Start）：
- 在 `_reset_idx` 中标记刚重置的环境（而非立刻清零）。
- 在下一次 `_get_observations` 中，当前帧的真实观测计算完成后，
  用此值填充整个历史 Buffer，消除异常零值。

## Checklist

- ✅ ~~在 `__init__` 中添加 `_just_reset_mask` 布尔张量~~ <!-- id: 0 -->
- ✅ ~~修改 `_reset_idx`：改为设置 mask 而非归零 Buffer~~ <!-- id: 1 -->
- ✅ ~~修改 `_get_observations`：加入 warm-start 逻辑~~ <!-- id: 2 -->
- ✅ ~~更新 DEV_LOG.md~~ <!-- id: 3 -->
