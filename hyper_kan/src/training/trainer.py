"""
训练器 - 支持暂停恢复机制
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    # PyTorch 2.0+ 新API
    from torch.amp import autocast, GradScaler
except ImportError:
    # PyTorch 1.x 旧API
    from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time
from typing import Dict, Any, Optional, List
import logging
from tqdm import tqdm

from ..utils.metrics import compute_metrics
from ..utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from ..utils.visualization import plot_loss_curve
from ..utils.logger import setup_logger

logger = logging.getLogger("HyperGKAN")

# ANSI颜色代码
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'


class Trainer:
    """
    HyperGKAN 训练器
    
    功能:
    - 训练循环
    - 验证
    - Early Stopping
    - Checkpoint保存/加载
    - 暂停/恢复训练
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[Any],
                 loss_fn: nn.Module,
                 H_nei: torch.Tensor,
                 H_sem: torch.Tensor,
                 W_nei: Optional[torch.Tensor],
                 W_sem: Optional[torch.Tensor],
                 device: str,
                 config: Dict[str, Any],
                 preprocessor=None,
                 output_dir: Optional[str] = None):
        """
        Args:
            model: HyperGKAN模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_fn: 损失函数
            H_nei: 邻域超图
            H_sem: 语义超图
            W_nei: 邻域权重
            W_sem: 语义权重
            device: 设备
            config: 配置字典
        """
        # DataParallel 检测：DataParallel 模型已经在 GPU 上，不需要再 .to(device)
        self._is_data_parallel = isinstance(model, nn.DataParallel)
        self.model = model if self._is_data_parallel else model.to(device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        
        # 超图数据
        if self._is_data_parallel:
            # DataParallel 模式: 超图已注册为模型 buffer，无需额外存储
            self.H_nei = None
            self.H_sem = None
            self.W_nei = None
            self.W_sem = None
            logger.info("DataParallel mode: 超图矩阵已作为模型 buffer，forward 时从 buffer 读取")
        else:
            # 单卡模式: 超图作为独立 tensor 存储
            self.H_nei = H_nei.to(device)
            self.H_sem = H_sem.to(device)
            self.W_nei = W_nei.to(device) if W_nei is not None else None
            self.W_sem = W_sem.to(device) if W_sem is not None else None
        
        self.device = device
        self.config = config

        # 训练配置
        self.epochs = config['training']['epochs']
        self.grad_clip = config['training']['grad_clip']
        self.use_amp = config['training']['use_amp']
        self.accumulation_steps = config['training']['accumulation_steps']

        # Early stopping
        self.patience = config['training']['early_stopping']['patience']
        self.min_delta = config['training']['early_stopping']['min_delta']

        # 定时暂停配置
        self.time_limit_minutes = config['training'].get('time_limit_minutes', None)
        self.training_start_time = None

        # 输出配置（使用传入的 output_dir 或创建新的）
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(
                config['output']['base_dir'],
                f"{time.strftime('%Y%m%d_%H%M%S')}_{config['meta']['element']}"
            )
        os.makedirs(self.output_dir, exist_ok=True)

        # checkpoint 目录（在训练输出目录下）
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # AMP scaler (兼容PyTorch 2.0+)
        if self.use_amp:
            try:
                # PyTorch 2.0+ 新API
                self.scaler = GradScaler(device='cuda')
            except TypeError:
                # PyTorch 1.x 旧API
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # 预处理器（用于反标准化）
        self.preprocessor = preprocessor

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        # 历史记录
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
        
        logger.info("Trainer initialized")
        logger.info(f"Output directory: {self.output_dir}")
        if self.time_limit_minutes:
            logger.info(f"Time limit: {self.time_limit_minutes} minutes")
    
    def check_time_limit(self) -> bool:
        """
        检查是否超过时间限制

        Returns:
            True: 超时，应该暂停
            False: 未超时，继续训练
        """
        if self.time_limit_minutes is None or self.training_start_time is None:
            return False

        elapsed_minutes = (time.time() - self.training_start_time) / 60.0

        if elapsed_minutes >= self.time_limit_minutes:
            logger.warning(f"⏰ Time limit reached: {elapsed_minutes:.1f} / {self.time_limit_minutes} minutes")
            return True

        return False

    def check_pause_flag(self) -> bool:
        """
        检查是否存在暂停标志文件

        Returns:
            True: 存在暂停标志，应该暂停
            False: 不存在暂停标志，继续训练
        """
        pause_flag = os.path.join(self.output_dir, '.pause')

        if not os.path.exists(pause_flag):
            return False

        # 读取暂停时间戳
        try:
            with open(pause_flag, 'r') as f:
                pause_time = float(f.read().strip())

            # 检查是否到达暂停时间
            if time.time() >= pause_time:
                logger.info("⏸️  Pause signal detected via .pause flag")
                return True
            else:
                # 未到达暂停时间，继续训练
                remaining_minutes = (pause_time - time.time()) / 60.0
                logger.debug(f"Pause flag exists, but waiting {remaining_minutes:.1f} more minutes")
                return False

        except Exception as e:
            logger.warning(f"Failed to read pause flag: {e}")
            return False

    def clear_pause_flag(self):
        """清除暂停标志文件"""
        pause_flag = os.path.join(self.output_dir, '.pause')

        if os.path.exists(pause_flag):
            try:
                os.remove(pause_flag)
                logger.info("✓ Pause flag cleared")
            except Exception as e:
                logger.warning(f"Failed to clear pause flag: {e}")
    
    def get_elapsed_time_info(self) -> str:
        """获取已用时间信息"""
        if self.training_start_time is None:
            return "N/A"
        
        elapsed_seconds = time.time() - self.training_start_time
        elapsed_minutes = elapsed_seconds / 60.0
        elapsed_hours = elapsed_minutes / 60.0
        
        if self.time_limit_minutes:
            remaining_minutes = self.time_limit_minutes - elapsed_minutes
            return (f"{elapsed_minutes:.1f}/{self.time_limit_minutes} min "
                   f"(remaining: {remaining_minutes:.1f} min)")
        else:
            if elapsed_hours >= 1:
                return f"{elapsed_hours:.2f} hours"
            else:
                return f"{elapsed_minutes:.1f} minutes"
    
    def _model_forward(self, x: torch.Tensor, output_length: int) -> torch.Tensor:
        """
        模型前向传播（兼容 DataParallel 和单卡模式）
        
        DataParallel 模式: 超图矩阵已在模型 buffer 中, 只传 x
        单卡模式: 超图矩阵作为参数传入
        """
        if self._is_data_parallel:
            # DataParallel: 只传 x (tensor, 沿batch维切分)
            # output_length 是 int, DataParallel 会自动广播
            return self.model(x, output_length=output_length)
        else:
            return self.model(x, self.H_nei, self.H_sem,
                            self.W_nei, self.W_sem,
                            output_length=output_length)
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader,
                   desc=f"Epoch {self.current_epoch+1}/{self.epochs}",
                   colour='green')
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 第一个batch添加调试信息
            if batch_idx == 0 and self.current_epoch == 0:
                logger.info("Starting first forward pass (may take 1-2 minutes for CUDA compilation)...")
            
            x = batch['x'].to(self.device)  # (B, T_in, N, F)
            y = batch['y'].to(self.device)  # (B, T_out, N, F)
            
            # 前向传播（使用 _model_forward 兼容 DataParallel）
            if self.use_amp:
                # 兼容PyTorch 2.0+
                try:
                    autocast_context = autocast(device_type='cuda', dtype=torch.float16)
                except TypeError:
                    autocast_context = autocast()
                
                with autocast_context:
                    pred = self._model_forward(x, output_length=y.shape[1])
                    loss = self.loss_fn(pred, y)
                    loss = loss / self.accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
            else:
                pred = self._model_forward(x, output_length=y.shape[1])
                loss = self.loss_fn(pred, y)
                loss = loss / self.accumulation_steps
                loss.backward()
            
            # 第一个batch完成提示
            if batch_idx == 0 and self.current_epoch == 0:
                logger.info("First forward pass completed successfully!")
            
            # 梯度累积
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # 梯度裁剪
                if self.grad_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # 优化器步进
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # NaN安全检查：跳过NaN loss的batch，避免污染整个epoch
            batch_loss_val = loss.item() * self.accumulation_steps
            if not np.isnan(batch_loss_val):
                epoch_loss += batch_loss_val
                num_batches += 1
            else:
                if not hasattr(self, '_nan_warned'):
                    logger.warning("Detected NaN loss in training batch, skipping. "
                                   "This may indicate numerical issues.")
                    self._nan_warned = True
            
            pbar.set_postfix({'loss': f'{batch_loss_val:.4f}'})

        # 显式关闭进度条，确保后续输出不被覆盖
        pbar.close()
        logger.info("")  # 添加空行分隔

        avg_loss = epoch_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """验证（使用 AMP 加速推理，与训练保持一致）"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)

                # 验证也使用 AMP 加速（与训练一致，避免验证比训练慢数倍）
                if self.use_amp:
                    try:
                        autocast_context = autocast(device_type='cuda', dtype=torch.float16)
                    except TypeError:
                        autocast_context = autocast()

                    with autocast_context:
                        pred = self._model_forward(x, output_length=y.shape[1])
                        loss = self.loss_fn(pred, y)
                else:
                    pred = self._model_forward(x, output_length=y.shape[1])
                    loss = self.loss_fn(pred, y)

                val_loss += loss.item()
                num_batches += 1

                all_preds.append(pred.float().cpu())
                all_targets.append(y.float().cpu())

        avg_val_loss = val_loss / num_batches

        # 计算其他指标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 如果有预处理器，反标准化数据后再计算指标
        if self.preprocessor is not None and self.preprocessor.fitted:
            all_preds_np = all_preds.numpy()
            all_targets_np = all_targets.numpy()

            # 反标准化
            all_preds_inv = self.preprocessor.inverse_transform(all_preds_np)
            all_targets_inv = self.preprocessor.inverse_transform(all_targets_np)

            # 转回tensor
            all_preds = torch.from_numpy(all_preds_inv).float()
            all_targets = torch.from_numpy(all_targets_inv).float()

        metrics = compute_metrics(
            all_preds,
            all_targets,
            metrics=self.config['evaluation']['metrics']
        )

        return {'loss': avg_val_loss, **metrics}
    
    def train(self, resume_from: Optional[str] = None):
        """
        完整训练流程
        
        Args:
            resume_from: 恢复训练的checkpoint路径
        """
        # 恢复训练
        if resume_from is not None:
            self.resume_training(resume_from)
        
        # 记录训练开始时间
        self.training_start_time = time.time()
        
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.epochs}")
        logger.info(f"Device: {self.device}")
        
        if self.time_limit_minutes:
            logger.info(f"⏰ Training will auto-pause after {self.time_limit_minutes} minutes")
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_metrics = self.validate()
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)
            self.val_metrics_history.append(val_metrics)

            # 日志 - 输出到终端和日志文件
            print()  # 立即输出到终端
            print("=" * 60)
            print(f"Epoch {epoch+1}/{self.epochs}")
            print("=" * 60)
            print(f"  Train Loss:      {train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            for metric, value in val_metrics.items():
                if metric != 'loss':
                    print(f"  Val {metric.upper():<6}:       {value:.4f}")

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning Rate:   {current_lr:.6f}")

            # 显示已用时间
            elapsed_info = self.get_elapsed_time_info()
            print(f"  Elapsed Time:    {elapsed_info}")

            # 保存checkpoint
            is_best = val_loss < self.best_val_loss - self.min_delta

            if is_best:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                # 动态精度：改进值小时显示更多小数位
                imp_str = f"{improvement:.4f}" if improvement < 0.001 else f"{improvement:.3f}"
                print(f"  ✨ NEW BEST MODEL! Val Loss: {val_loss:.4f} (-{imp_str})")
            else:
                self.epochs_no_improve += 1

            print("=" * 60)

            # 同步到日志文件
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            logger.info("=" * 60)
            logger.info(f"  Train Loss:      {train_loss:.4f}")
            logger.info(f"  Validation Loss: {val_loss:.4f}")
            for metric, value in val_metrics.items():
                if metric != 'loss':
                    logger.info(f"  Val {metric.upper():<6}:       {value:.4f}")

            if self.scheduler is not None:
                logger.info(f"  Learning Rate:   {current_lr:.6f}")

            logger.info(f"  Elapsed Time:    {elapsed_info}")

            if is_best:
                imp_str = f"{improvement:.4f}" if improvement < 0.001 else f"{improvement:.3f}"
                logger.info(f"  ✨ NEW BEST MODEL! Val Loss: {val_loss:.4f} (-{imp_str})")

            logger.info("=" * 60)
            
            # 保存checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # 绘制损失曲线
            if (epoch + 1) % 10 == 0:
                plot_loss_curve(
                    self.train_losses,
                    self.val_losses,
                    os.path.join(self.output_dir, 'loss_curve.png'),
                    title=f"Loss Curve - {self.config['meta']['element']}"
                )
            
            # 检查时间限制
            if self.check_time_limit():
                logger.warning("=" * 60)
                logger.warning("⏰ TIME LIMIT REACHED - Auto-pausing training")
                logger.warning("=" * 60)
                logger.warning(f"Completed {epoch+1} epochs")
                logger.warning(f"Current best validation loss: {self.best_val_loss:.4f}")
                logger.warning("Saving checkpoint before pausing...")

                # 保存当前状态
                self.save_checkpoint(is_best=False)

                logger.warning("=" * 60)
                logger.warning("Checkpoint saved successfully!")
                logger.warning("To resume training, use:")
                logger.warning(f"  python train.py --resume {os.path.join(self.checkpoint_dir, 'last.pt')}")
                logger.warning("Or:")
                logger.warning("  python pause_resume/resume.py")
                logger.warning("=" * 60)
                break

            # 检查暂停标志文件
            if self.check_pause_flag():
                logger.warning("=" * 60)
                logger.warning("⏸️  PAUSE SIGNAL RECEIVED - Pausing training")
                logger.warning("=" * 60)
                logger.warning(f"Completed {epoch+1} epochs")
                logger.warning(f"Current best validation loss: {self.best_val_loss:.4f}")
                logger.warning("Saving checkpoint before pausing...")

                # 保存当前状态
                self.save_checkpoint(is_best=False)

                # 清除暂停标志
                self.clear_pause_flag()

                logger.warning("=" * 60)
                logger.warning("Checkpoint saved successfully!")
                logger.warning("To resume training, use:")
                logger.warning(f"  python train.py --resume {os.path.join(self.checkpoint_dir, 'last.pt')}")
                logger.warning("Or:")
                logger.warning("  python pause_resume/resume.py")
                logger.warning("=" * 60)
                break
            
            # Early stopping
            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # 最终保存
        self.save_checkpoint(is_best=False)
        plot_loss_curve(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curve_final.png'),
            title=f"Final Loss Curve - {self.config['meta']['element']}"
        )
    
    def save_checkpoint(self, is_best: bool = False):
        """保存checkpoint到统一的 outputs/checkpoints 目录
        
        注意：DataParallel 包装的模型需要通过 .module 获取原始模型的 state_dict，
        这样保存的 checkpoint 在加载时不需要关心是否使用了多卡训练。
        """
        # 兼容 DataParallel: 提取原始模型的 state_dict
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        state = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics_history': self.val_metrics_history,
            'config': self.config
        }

        # 使用统一的 checkpoint 目录
        save_checkpoint(
            state,
            self.checkpoint_dir,
            filename=f"checkpoint_epoch_{self.current_epoch+1}.pt",
            is_best=is_best
        )
    
    def resume_training(self, checkpoint_path: str):
        """恢复训练（兼容 DataParallel）"""
        logger.info(f"Resuming training from {checkpoint_path}")
        
        # DataParallel 兼容：加载时使用原始模型
        model_for_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        checkpoint = load_checkpoint(
            checkpoint_path,
            model_for_load,
            self.optimizer,
            self.scheduler,
            self.device
        )
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics_history = checkpoint.get('val_metrics_history', [])
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
        logger.info(f"Best validation loss so far: {self.best_val_loss:.4f}")