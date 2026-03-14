"""
HyperGKAN 一键训练+预测
用法: python main.py --config configs/config.yaml --gpu 0

流程:
  1. 训练模型 → 输出到 outputs/{时间戳}_{数据集}/
  2. 自动定位 best_model.pt
  3. 在同一输出目录中完成预测与评估
"""
import os
import sys
import glob
import subprocess
import argparse


def find_latest_output_dir(base_dir: str) -> str:
    """
    扫描 outputs/ 下所有子目录，返回修改时间最新的那个
    （即刚刚训练完成生成的目录）
    """
    if not os.path.isdir(base_dir):
        return None

    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not subdirs:
        return None

    # 按修改时间降序，取最新
    subdirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    return subdirs[0]


def find_best_checkpoint(output_dir: str) -> str:
    """
    在训练输出目录中查找 best_model.pt
    优先级: best_model.pt > last.pt > 最新 checkpoint_epoch_*.pt
    """
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    # 优先 best_model.pt
    best = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.isfile(best):
        return best

    # 其次 last.pt
    last = os.path.join(ckpt_dir, "last.pt")
    if os.path.isfile(last):
        return last

    # 最后按文件名排序取最新
    pts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pt")))
    if pts:
        return pts[-1]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="HyperGKAN 一键训练 + 预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练 + 预测（最常用）
  python main.py --config configs/config.yaml --gpu 0

  # 仅训练
  python main.py --config configs/config.yaml --gpu 0 --skip_predict

  # 仅预测（使用最新训练结果）
  python main.py --config configs/config.yaml --gpu 0 --skip_train

  # 恢复训练后继续预测
  python main.py --config configs/config.yaml --gpu 0 --resume outputs/xxx/checkpoints/last.pt
        """
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="配置文件路径 (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="指定 GPU 编号 (0-5)"
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="多GPU模式: 逗号分隔 (如 0,1,2)"
    )
    parser.add_argument(
        "--multi_gpu", action="store_true", default=False,
        help="启用 DataParallel 多卡并行"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="恢复训练的 checkpoint 路径"
    )
    parser.add_argument(
        "--skip_train", action="store_true", default=False,
        help="跳过训练，仅对最新输出执行预测"
    )
    parser.add_argument(
        "--skip_predict", action="store_true", default=False,
        help="跳过预测，仅执行训练"
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable  # 使用当前 Python 解释器（兼容 conda 环境）
    outputs_base = os.path.join(script_dir, "outputs")

    # ========== 阶段 1: 训练 ==========
    if not args.skip_train:
        print("=" * 60)
        print("  阶段 1/2: 训练")
        print("=" * 60)

        train_cmd = [python, os.path.join(script_dir, "train.py"),
                     "--config", args.config]

        if args.gpu is not None:
            train_cmd.extend(["--gpu", str(args.gpu)])
        if args.gpus is not None:
            train_cmd.extend(["--gpus", args.gpus])
        if args.multi_gpu:
            train_cmd.append("--multi_gpu")
        if args.resume:
            train_cmd.extend(["--resume", args.resume])

        print(f"  命令: {' '.join(train_cmd)}")
        print()

        ret = subprocess.run(train_cmd)
        if ret.returncode != 0:
            print(f"\n❌ 训练失败 (exit code {ret.returncode})，跳过预测。")
            sys.exit(ret.returncode)

        print()
        print("✅ 训练完成!")
        print()

    # ========== 定位输出目录和 checkpoint ==========
    if args.resume and args.skip_train:
        # 跳过训练 + 指定了 resume checkpoint → 从中推断输出目录
        ckpt_abs = os.path.abspath(args.resume)
        if "checkpoints" in ckpt_abs:
            output_dir = os.path.dirname(os.path.dirname(ckpt_abs))
        else:
            output_dir = find_latest_output_dir(outputs_base)
        checkpoint_path = ckpt_abs
    else:
        # 正常流程: 查找最新的输出目录
        output_dir = find_latest_output_dir(outputs_base)
        if output_dir is None:
            print("❌ 未找到任何训练输出目录。")
            sys.exit(1)
        checkpoint_path = find_best_checkpoint(output_dir)

    if checkpoint_path is None:
        print(f"❌ 在 {output_dir} 中未找到任何 checkpoint 文件。")
        sys.exit(1)

    print(f"  输出目录:   {output_dir}")
    print(f"  Checkpoint: {checkpoint_path}")

    # ========== 阶段 2: 预测 ==========
    if not args.skip_predict:
        print()
        print("=" * 60)
        print("  阶段 2/2: 预测与评估")
        print("=" * 60)

        predict_cmd = [python, os.path.join(script_dir, "predict.py"),
                       "--config", args.config,
                       "--checkpoint", checkpoint_path,
                       "--output", output_dir]

        if args.gpu is not None:
            predict_cmd.extend(["--gpu", str(args.gpu)])

        print(f"  命令: {' '.join(predict_cmd)}")
        print()

        ret = subprocess.run(predict_cmd)
        if ret.returncode != 0:
            print(f"\n❌ 预测失败 (exit code {ret.returncode})")
            sys.exit(ret.returncode)

        print()
        print("✅ 预测完成!")

    # ========== 汇总 ==========
    print()
    print("=" * 60)
    print("  全部完成!")
    print("=" * 60)
    print(f"  所有结果保存在: {output_dir}")
    print()
    print("  目录内容:")
    if os.path.isdir(output_dir):
        for item in sorted(os.listdir(output_dir)):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                print(f"    📁 {item}/")
            else:
                size_kb = os.path.getsize(item_path) / 1024
                if size_kb > 1024:
                    print(f"    📄 {item}  ({size_kb/1024:.1f} MB)")
                else:
                    print(f"    📄 {item}  ({size_kb:.1f} KB)")
    print()


if __name__ == "__main__":
    main()
