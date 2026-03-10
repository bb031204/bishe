"""
HyperGKAN 主程序
统一入口: 训练、预测、可视化
"""
import argparse
import subprocess
import sys


def run_train(args):
    """运行训练"""
    cmd = ['python', 'train.py', '--config', args.config]

    if args.resume:
        cmd.extend(['--resume', args.resume])

    if args.gpu is not None:
        cmd.extend(['--gpu', str(args.gpu)])

    subprocess.run(cmd)


def run_predict(args):
    """运行预测"""
    if not args.checkpoint:
        print("Error: --checkpoint is required for prediction")
        sys.exit(1)

    cmd = ['python', 'predict.py', '--config', args.config, '--checkpoint', args.checkpoint]

    if args.output:
        cmd.extend(['--output', args.output])

    if args.gpu is not None:
        cmd.extend(['--gpu', str(args.gpu)])

    subprocess.run(cmd)


def run_visualize(args):
    """运行可视化"""
    print("Visualization module not implemented yet")
    # 可以添加可视化脚本


def main():
    parser = argparse.ArgumentParser(description="HyperGKAN - Main Entry Point")
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    train_parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device number to use (0-5)'
    )
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    predict_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    predict_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory'
    )
    predict_parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device number to use (0-5)'
    )
    
    # 可视化命令
    viz_parser = subparsers.add_parser('visualize', help='Visualize results')
    viz_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to predictions file'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_train(args)
    elif args.command == 'predict':
        run_predict(args)
    elif args.command == 'visualize':
        run_visualize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
