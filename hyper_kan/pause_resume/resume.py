"""
resume.py - 恢复暂停的训练

使用方法：
    # 自动检测最新训练并恢复
    python pause_resume/resume.py

    # 指定 checkpoint 路径恢复
    python pause_resume/resume.py --checkpoint outputs/20260127_120000_Temperature/checkpoints/checkpoint_epoch_15.pt

    # 恢复后50分钟自动暂停
    python pause_resume/resume.py --resume-time 50

    # 仅显示 checkpoint 信息，不启动训练
    python pause_resume/resume.py --info

功能：
    1. 从上次保存的 checkpoint 继续训练
    2. 完全恢复训练状态（模型、优化器、调度器、epoch、历史记录）
    3. 训练质量与不间断训练完全一致
    4. 支持定时暂停功能

参数：
    --checkpoint: 指定 checkpoint 文件路径
    --config: 配置文件路径（默认使用训练时保存的config）
    --resume-time: 恢复后多少分钟自动暂停（0表示不自动暂停）
    --info: 仅显示 checkpoint 信息，不启动训练

恢复的状态包括：
    - 模型参数
    - 优化器状态（包括动量）
    - 学习率调度器状态
    - 当前 epoch
    - 训练/验证损失历史
    - 最佳模型指标
"""

import os
import sys
import argparse
import yaml
import torch
import time

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def find_latest_training_dir(base_dir: str = "outputs"):
    """
    查找最新的训练目录

    Args:
        base_dir: outputs基础目录

    Returns:
        最新的训练目录路径，如果找不到返回None
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(script_dir, base_dir)

    if not os.path.exists(outputs_dir):
        return None

    # 查找所有时间戳目录
    timestamp_dirs = []
    for dir_name in os.listdir(outputs_dir):
        if dir_name.count('_') >= 2:  # 格式: YYYYMMDD_HHMMSS_Element
            try:
                dir_path = os.path.join(outputs_dir, dir_name)
                if os.path.isdir(dir_path):
                    timestamp_dirs.append((dir_path, dir_name))
            except:
                pass

    if not timestamp_dirs:
        return None

    # 按时间戳排序，取最新的
    timestamp_dirs.sort(key=lambda x: x[1], reverse=True)
    return timestamp_dirs[0][0]


def find_saved_config(training_dir: str) -> str:
    """
    在训练目录中查找保存的config.yaml

    Args:
        training_dir: 训练目录路径

    Returns:
        config文件路径，如果找不到返回None
    """
    config_path = os.path.join(training_dir, 'config.yaml')

    if os.path.exists(config_path):
        return config_path

    return None


def get_latest_checkpoint_from_dir(checkpoint_dir: str) -> str:
    """
    从目录中查找最新的checkpoint文件

    Args:
        checkpoint_dir: checkpoint目录路径

    Returns:
        最新checkpoint路径，找不到返回None
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # 查找所有 .pt 文件
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pt'):
            filepath = os.path.join(checkpoint_dir, filename)
            mtime = os.path.getmtime(filepath)
            checkpoints.append((filepath, mtime, filename))

    if not checkpoints:
        return None

    # 优先选择 last.pt，否则按修改时间排序
    for cp in checkpoints:
        if cp[2] == 'last.pt':
            return cp[0]

    # 按 modify time 排序，返回最新的
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints[0][0]


def print_checkpoint_info(checkpoint_path: str):
    """打印 checkpoint 信息"""
    print("\n" + "=" * 60)
    print("Checkpoint 信息")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"文件: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")

    # Try to get best_val_loss, may be named differently in different versions
    if 'best_val_loss' in checkpoint:
        print(f"最佳验证 Loss: {checkpoint['best_val_loss']:.4f}")
    elif 'best_val_mae' in checkpoint:
        print(f"最佳验证 MAE: {checkpoint['best_val_mae']:.4f}")

    print(f"训练历史长度: {len(checkpoint.get('train_losses', []))} epochs")

    # 检查配置
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        total_epochs = cfg.get('training', {}).get('epochs', 100)
        current_epoch = checkpoint.get('epoch', 0)
        remaining = total_epochs - current_epoch
        print(f"总计划轮数: {total_epochs}")
        print(f"剩余轮数: {remaining}")

    print("=" * 60 + "\n")


def setup_auto_pause(out_dir: str, resume_minutes: float):
    """
    设置自动暂停

    Args:
        out_dir: 输出目录
        resume_minutes: 恢复后多少分钟自动暂停
    """
    if resume_minutes <= 0:
        return

    pause_flag = os.path.join(out_dir, '.pause')

    # 检查是否已有暂停标志
    if os.path.exists(pause_flag):
        print(f"警告: 已存在暂停标志，将覆盖")
        try:
            os.remove(pause_flag)
        except:
            pass

    # 写入定时暂停
    target_time = time.time() + resume_minutes * 60
    try:
        with open(pause_flag, 'w') as f:
            f.write(str(target_time))

        hours = int(resume_minutes // 60)
        mins = int(resume_minutes % 60)
        if hours > 0:
            time_str = f"{hours}小时{mins}分钟"
        else:
            time_str = f"{mins}分钟"

        print(f"✓ 已设置自动暂停: {time_str} 后")
        print(f"  预计暂停时间: {time.ctime(target_time)}")
    except Exception as e:
        print(f"✗ 设置自动暂停失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='HyperGKAN - 训练恢复工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pause_resume/resume.py                                    # 自动恢复最新训练
  python pause_resume/resume.py --checkpoint path/to/checkpoint.pt # 指定 checkpoint
  python pause_resume/resume.py --resume-time 50                   # 50分钟后自动暂停
  python pause_resume/resume.py --info                             # 仅显示信息
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint文件路径。如果不指定，自动使用最新的训练结果。'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config文件路径。如果不指定，自动使用训练时保存的config。'
    )
    parser.add_argument(
        '--resume-time',
        type=float,
        default=0,
        help='恢复后多少分钟自动暂停（0表示不自动暂停）'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='仅显示 checkpoint 信息，不启动训练'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HyperGKAN - 训练恢复工具")
    print("=" * 60)

    # 确定checkpoint路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"✗ 指定的 checkpoint 不存在: {checkpoint_path}")
            return 1
        print(f"✓ 使用指定的 checkpoint: {checkpoint_path}")
    else:
        print("🔍 自动查找最新checkpoint...")
        # 查找最新训练目录
        latest_dir = find_latest_training_dir()

        if not latest_dir:
            print(f"✗ 未找到可恢复的训练目录")
            print(f"  请确保已经完成过至少一次训练")
            return 1

        print(f"✓ 最新训练目录: {latest_dir}")

        # 查找checkpoint
        checkpoint_dir = os.path.join(latest_dir, 'checkpoints')
        checkpoint_path = get_latest_checkpoint_from_dir(checkpoint_dir)

        if not checkpoint_path:
            print(f"✗ 在 {checkpoint_dir} 中找不到checkpoint")
            return 1

        print(f"✓ 找到Checkpoint: {checkpoint_path}")

    # 打印checkpoint信息
    print_checkpoint_info(checkpoint_path)

    if args.info:
        # 仅显示信息
        return 0

    # 确定config路径
    if args.config:
        config_path = args.config
        print(f"✓ 使用指定的Config: {config_path}")
    else:
        # 从checkpoint路径推断训练目录
        checkpoint_dir = os.path.dirname(checkpoint_path)
        training_dir = os.path.dirname(checkpoint_dir)

        # 查找保存的config
        config_path = find_saved_config(training_dir)

        if not config_path or not os.path.exists(config_path):
            print(f"⚠️  警告: 找不到保存的config.yaml")
            print(f"   使用默认config: configs/config.yaml")
            config_path = 'configs/config.yaml'
        else:
            print(f"✓ 使用训练时保存的Config: {config_path}")

    # 验证config存在
    if not os.path.exists(config_path):
        print(f"✗ Config文件不存在: {config_path}")
        return 1

    # 设置自动暂停
    if args.resume_time > 0:
        # 获取输出目录（从checkpoint路径推断训练目录）
        checkpoint_dir = os.path.dirname(checkpoint_path)
        out_dir = os.path.dirname(checkpoint_dir)

        print("\n" + "=" * 60)
        print("设置自动暂停")
        print("=" * 60)
        setup_auto_pause(out_dir, args.resume_time)

    # 启动恢复训练
    print("\n" + "=" * 60)
    print("准备恢复训练...")
    print("=" * 60)
    print("\n将从 checkpoint 继续训练，状态包括:")
    print("  ✓ 模型参数")
    print("  ✓ 优化器状态（动量等）")
    print("  ✓ 学习率调度器状态")
    print("  ✓ 训练历史记录")
    print("  ✓ 最佳模型记录")

    if args.resume_time > 0:
        print(f"\n⏰ 将在 {args.resume_time} 分钟后自动暂停")

    print("\n训练质量将与不间断训练完全一致\n")

    # 导入训练模块并调用
    import subprocess

    cmd = [
        sys.executable,
        os.path.join(project_root, 'train.py'),
        '--config', config_path,
        '--resume', checkpoint_path
    ]

    print("=" * 60)
    print("启动训练命令:")
    print(" ".join(cmd))
    print("=" * 60)
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("恢复训练完成!")
        print("=" * 60)
        return 0
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 恢复训练出错: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
