# 训练暂停与恢复功能

## ⚡ 快速开始

**重要：请确保在项目根目录 `d:/bishe/code/hyper_kan` 下执行命令！**

```bash
# 进入项目目录
cd d:/bishe/code/hyper_kan

# 暂停训练（60分钟后）
python d:/bishe/code/hyper_kan/pause_resume/pause.py --pause-time 60

# 立即暂停
python d:/bishe/code/hyper_kan/pause_resume/pause.py

# 恢复训练
python d:/bishe/code/hyper_kan/pause_resume/resume.py

# 查看checkpoint信息
python pause_resume/resume.py --info
```

**或者从任意位置使用批处理文件（在 `d:/bishe` 目录下）：**
```bash
pause_train.bat 60     # 暂停
resume_train.bat       # 恢复
```

---

## 📋 功能说明

### 1. 定时暂停训练 (`pause.py`)

在指定时间后完成当前epoch并自动暂停，保存训练状态。

**用法（在项目根目录 d:/bishe/code/hyper_kan 下）：**
```bash
# 立即暂停（完成当前epoch后）
python pause_resume/pause.py

# 5分钟后暂停
python pause_resume/pause.py --pause-time 5

# 60分钟后暂停
python pause_resume/pause.py --pause-time 60

# 2小时后暂停
python pause_resume/pause.py --pause-time 120
```

**用法（从 d:/bishe 目录使用批处理文件）：**
```bash
# 暂停训练
pause_train.bat 60

# 恢复训练
resume_train.bat
```

**工作原理：**
1. 创建 `.pause` 标志文件，包含目标时间戳
2. 训练器在每个 epoch 后检查该文件
3. 到达指定时间后，完成当前 epoch
4. 自动保存 checkpoint 并退出
5. 自动清除 `.pause` 标志文件

**暂停后保存的文件：**
```
outputs/20260127_120000_Temperature/
├── checkpoints/
│   ├── checkpoint_epoch_15.pt    # 暂停时的epoch
│   └── best_model.pt             # 最佳模型
├── preprocessor.pkl              # 数据预处理器
└── config.yaml                   # 训练配置
```

---

### 2. 恢复训练 (`resume.py`)

自动查找最新的训练结果并恢复训练。

**用法（在项目根目录 d:/bishe/code/hyper_kan 下）：**
```bash
# 自动恢复最新训练
python pause_resume/resume.py

# 指定checkpoint恢复
python pause_resume/resume.py --checkpoint outputs/20260127_120000_Temperature/checkpoints/checkpoint_epoch_15.pt

# 恢复后50分钟自动暂停
python pause_resume/resume.py --resume-time 50

# 仅查看checkpoint信息（不启动训练）
python pause_resume/resume.py --info

# 指定config恢复
python pause_resume/resume.py --config configs/custom_config.yaml
```

**恢复流程：**
1. 自动查找最新的训练目录
2. 查找最新的 checkpoint 文件（优先 `last.pt`）
3. 自动加载训练时保存的 `config.yaml`（确保配置一致）
4. 显示训练信息（epoch、best_val_loss）
5. 启动训练

---

## 🔄 完整工作流程示例

### 场景1：长时间训练分批进行

```bash
# 在项目根目录 d:/bishe/code/hyper_kan 下执行

# 第1天：训练2小时
python pause_resume/pause.py --pause-time 120

# ... 训练进行中，自动暂停 ...

# 第2天：继续训练
python pause_resume/resume.py

# 第3天：再次继续
python pause_resume/resume.py
```

或从 d:/bishe 目录使用批处理文件：
```bash
# 第1天：训练2小时
pause_train.bat 120

# 第2天：继续训练
resume_train.bat
```

### 场景2：快速测试调整

```bash
# 在项目根目录下
python pause_resume/pause.py --pause-time 30

# 检查结果后决定是否继续
# 如果继续：
python pause_resume/resume.py
```

### 场景3：指定特定checkpoint恢复

```bash
# 恢复到某个特定的checkpoint
python pause_resume/resume.py --checkpoint outputs/20260127_120000_Temperature/checkpoints/checkpoint_epoch_10.pt
```

### 场景4：仅查看训练进度

```bash
# 查看最新checkpoint信息，不启动训练
python pause_resume/resume.py --info
```

---

## ⚠️ 注意事项

### Config一致性

恢复训练时会**自动使用训练时保存的config.yaml**，确保：
- 数据路径相同
- 模型结构相同
- 超图参数相同

如果修改了config，需要明确指定：

```bash
python pause_resume/resume.py --config configs/new_config.yaml
```

### Checkpoint保存时机

- 每个epoch结束时保存 `checkpoint_epoch_<N>.pt`
- 验证loss改善时保存 `best_model.pt`
- 暂停时额外保存当前状态
- 暂停后自动清除 `.pause` 标志文件

### 预处理器 (preprocessor.pkl)

- 训练时自动保存数据标准化参数
- 恢复训练时自动加载
- 预测时也需要使用相同的预处理器

### 临时配置文件

**关于 `config_pause_*.yaml` 文件：**
- 这些是旧版本暂停功能创建的临时配置文件
- **可以安全删除**，不影响新版本的暂停/恢复功能
- 新版本使用 `.pause` 标志文件机制，不再需要临时配置文件

---

## 🛠️ 故障排除

### 问题1：找不到checkpoint

```
✗ 未找到可恢复的训练目录
```

**解决：**
- 确认已经完成过至少一次训练
- 检查 `outputs/` 目录是否存在
- 手动指定checkpoint路径

### 问题2：config不匹配

```
⚠️ 警告: 找不到保存的config.yaml
```

**解决：**
- 确认训练时 config.yaml 被正确保存
- 或手动指定config：`python pause_resume/resume.py --config configs/config.yaml`

### 问题3：暂停未生效

如果暂停后训练仍在继续：
1. 检查 `.pause` 文件是否已创建
2. 确认时间戳是否正确
3. 训练会在当前 epoch 完成后暂停，不会中断正在进行的 epoch

---

## 📊 输出示例

### pause.py 输出

```
============================================================
HyperGKAN - 训练暂停工具
============================================================

正在查找最新训练目录...
✓ 找到训练目录: outputs/20260127_120000_Temperature

============================================================
设置暂停信号
============================================================
  将在 60 分钟 后暂停
  预计暂停时间: Mon Jan 27 14:30:00 2026

============================================================
✓ 定时暂停信号已设置（60.0分钟后）
============================================================

训练将在当前 epoch 结束后:
  1. 保存 checkpoint
  2. 保存日志
  3. 清除暂停标志
  4. 优雅退出

恢复训练时，请运行:
  python pause_resume/resume.py

您现在可以:
  • 继续使用电脑（训练将在后台自动暂停）
  • 关闭终端/IDE
  • 让电脑休眠以节省能耗
============================================================
```

### resume.py 输出

```
============================================================
HyperGKAN - 训练恢复工具
============================================================
🔍 自动查找最新checkpoint...
✓ 最新训练目录: outputs/20260127_120000_Temperature
✓ 找到Checkpoint: outputs/20260127_120000_Temperature/checkpoints/checkpoint_epoch_15.pt

============================================================
Checkpoint 信息
============================================================
文件: outputs/20260127_120000_Temperature/checkpoints/checkpoint_epoch_15.pt
Epoch: 15
最佳验证 Loss: 1.2345
训练历史长度: 15 epochs
总计划轮数: 100
剩余轮数: 85
============================================================

✓ 使用训练时保存的Config: outputs/20260127_120000_Temperature/config.yaml

============================================================
准备恢复训练...
============================================================

将从 checkpoint 继续训练，状态包括:
  ✓ 模型参数
  ✓ 优化器状态（动量等）
  ✓ 学习率调度器状态
  ✓ 训练历史记录
  ✓ 最佳模型记录

训练质量将与不间断训练完全一致

============================================================
启动训练命令:
python train.py --config outputs/20260127_120000_Temperature/config.yaml --resume outputs/20260127_120000_Temperature/checkpoints/checkpoint_epoch_15.pt
============================================================
```

---

## 🔧 高级用法

### 查看checkpoint信息而不启动训练

```bash
python pause_resume/resume.py --info
```

### 恢复后自动暂停

```bash
# 恢复训练，并在50分钟后自动暂停
python pause_resume/resume.py --resume-time 50
```

### 指定特定的checkpoint

```bash
python pause_resume/resume.py --checkpoint outputs/20260127_120000_Temperature/checkpoints/checkpoint_epoch_100.pt
```

### 使用不同的配置文件

```bash
python pause_resume/resume.py --checkpoint <path> --config configs/config_modified.yaml
```

**⚠️ 注意：** 修改配置文件可能导致checkpoint不兼容。

---

## 🎯 工作原理

### 文件标志机制

新版本的暂停/恢复功能使用文件标志机制（`.pause` 文件）进行通信：

1. **暂停信号创建**：
   - `pause.py` 在训练目录创建 `.pause` 文件
   - 文件内容为目标暂停时间的时间戳
   - 支持立即暂停（当前时间）和定时暂停（未来时间）

2. **训练器检查**：
   - 训练器在每个 epoch 后检查 `.pause` 文件
   - 如果当前时间 >= 目标时间，触发暂停流程
   - 完成当前 epoch 后保存 checkpoint 并退出

3. **标志清除**：
   - 暂停完成后自动删除 `.pause` 文件
   - 确保下次训练不会误触发暂停

### 与旧版本的区别

| 特性 | 旧版本 | 新版本 |
|------|--------|--------|
| 暂停机制 | 创建临时配置文件 | 文件标志 (`.pause`) |
| Config文件 | `config_pause_*.yaml` | 使用训练时保存的 `config.yaml` |
| 清理 | 需手动删除临时配置 | 自动清除 `.pause` 标志 |
| 灵活性 | 有限 | 更灵活，支持立即/定时暂停 |
