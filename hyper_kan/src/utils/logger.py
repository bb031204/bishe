"""
日志配置
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional


class FlushStreamHandler(logging.StreamHandler):
    """自定义StreamHandler，确保每次日志后立即刷新"""
    def emit(self, record):
        super().emit(record)
        self.flush()


class FlushFileHandler(logging.FileHandler):
    """自定义FileHandler，确保每次日志后立即刷新"""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(name: str = "HyperGKAN",
                level: str = "INFO",
                log_dir: Optional[str] = None,
                output_dir: Optional[str] = None,
                console: bool = True,
                file: bool = True,
                append_mode: bool = False) -> logging.Logger:
    """
    设置日志器

    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志文件目录（已弃用，使用output_dir）
        output_dir: 输出目录（格式：outputs/YYYYMMDD_HHMMSS_Element）
        console: 是否输出到控制台
        file: 是否输出到文件
        append_mode: 是否追加模式（恢复训练时使用）

    Returns:
        配置好的logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 清除现有handlers
    logger.handlers.clear()

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台handler - 使用自定义handler确保立即刷新
    if console:
        console_handler = FlushStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件handler - 优先使用output_dir
    if file:
        # 确定日志保存目录
        if output_dir is not None:
            # 使用output_dir（outputs/YYYYMMDD_HHMMSS_Element）
            log_directory = output_dir
        elif log_dir is not None:
            # 兼容旧的log_dir参数
            log_directory = log_dir
        else:
            # 都没有提供，则不保存文件
            return logger

        os.makedirs(log_directory, exist_ok=True)

        # 恢复训练时，查找现有日志文件；否则创建新的
        if append_mode:
            # 查找现有的日志文件
            existing_logs = [f for f in os.listdir(log_directory) if f.startswith('train_') and f.endswith('.log')]
            if existing_logs:
                # 使用第一个找到的日志文件
                log_file = os.path.join(log_directory, existing_logs[0])
                mode = 'a'  # 追加模式
            else:
                # 没有找到，创建新的
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_directory, f"train_{timestamp}.log")
                mode = 'w'  # 写入模式
        else:
            # 创建新日志文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_directory, f"train_{timestamp}.log")
            mode = 'w'  # 写入模式

        file_handler = FlushFileHandler(log_file, encoding='utf-8', mode=mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if mode == 'a':
            logger.info(f"日志文件追加至: {log_file}")
        else:
            logger.info(f"日志文件保存至: {log_file}")

    return logger
