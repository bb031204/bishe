"""
气象预测专用执行器
- 仿照HyperGKAN日志格式输出 MAE/RMSE 指标
- 支持梯度累积 (Gradient Accumulation)
- 输出到指定目录
- 彩色日志输出
"""
import os
import time
import datetime
import numpy as np
import torch
from logging import getLogger
from libcity.executor.state_executor import StateExecutor
from libcity.utils import ensure_dir
from libcity.model import loss
from tqdm import tqdm
from libcity.utils.meteo_visualization import plot_predictions, plot_loss_curve


# ==================== ANSI颜色代码 ====================
class Colors:
    """ANSI颜色代码"""
    # 前景色
    BLACK = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    # 背景色
    BG_BLACK = '\033[100m'
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'

    # 样式
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    @staticmethod
    def color(text, color_code):
        """给文本添加颜色"""
        return f"{color_code}{text}{Colors.RESET}"

    @staticmethod
    def cyan(text):
        return Colors.color(text, Colors.CYAN)

    @staticmethod
    def green(text):
        return Colors.color(text, Colors.GREEN)

    @staticmethod
    def yellow(text):
        return Colors.color(text, Colors.YELLOW)

    @staticmethod
    def red(text):
        return Colors.color(text, Colors.RED)

    @staticmethod
    def blue(text):
        return Colors.color(text, Colors.BLUE)

    @staticmethod
    def magenta(text):
        return Colors.color(text, Colors.MAGENTA)

    @staticmethod
    def bold(text):
        return Colors.color(text, Colors.BOLD)


class MeteoExecutor(StateExecutor):
    """
    继承StateExecutor，增加：
    1. 每个epoch输出MAE/RMSE（仿照HyperGKAN日志格式）
    2. 梯度累积支持
    3. 测试集MAE/RMSE评估
    """

    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.accumulation_steps = self.config.get('accumulation_steps', 1)
        self.element = self.config.get('element', 'Unknown')
        self.model_name = self.config.get('model', 'Unknown')

        # 输出目录（使用相对路径，适配服务器）
        self.output_base = self.config.get('output_base_dir', './outputs')
        self.start_time = time.time()

    def _compute_metrics(self, dataloader, desc="Eval"):
        """
        在数据集上计算MAE和RMSE（与HyperGKAN完全一致：在原始尺度上计算）
        """
        self.model.eval()
        y_truths = []
        y_preds = []

        with torch.no_grad():
            for batch in dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())

        y_truths = np.concatenate(y_truths, axis=0)
        y_preds = np.concatenate(y_preds, axis=0)

        mae = np.mean(np.abs(y_preds - y_truths))
        rmse = np.sqrt(np.mean((y_preds - y_truths) ** 2))
        return mae, rmse

    def train(self, train_dataloader, eval_dataloader):
        """
        训练，带MAE/RMSE日志输出（仿照HyperGKAN格式）
        """
        self._logger.info('')
        self._logger.info(Colors.cyan('=' * 60))
        self._logger.info(Colors.bold(f'Starting training: {self.model_name} on {self.element}'))
        self._logger.info(Colors.cyan(f'  Total epochs: {self.epochs}'))
        self._logger.info(f'  Device: {self.device}')
        self._logger.info(f'  Batch size: {self.config.get("batch_size", "N/A")}')
        self._logger.info(f'  Gradient accumulation steps: {self.accumulation_steps}')
        eff_bs = self.config.get("batch_size", 16) * self.accumulation_steps
        self._logger.info(f'  Effective batch size: {eff_bs}')
        self._logger.info(Colors.cyan('=' * 60))

        min_val_loss = float('inf')
        best_val_mae = float('inf')
        best_val_rmse = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        train_loss_history = []
        val_loss_history = []
        self.start_time = time.time()

        for epoch_idx in range(self._epoch_num, self.epochs):
            # ---- 训练 ----
            start_time = time.time()
            losses = self._train_epoch_with_accumulation(train_dataloader, epoch_idx, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss = np.mean(losses)

            # ---- 验证 ----
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, self.loss_func)
            eval_time.append(time.time() - t2)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            # ---- 计算MAE/RMSE（在原始尺度上） ----
            val_mae, val_rmse = self._compute_metrics(eval_dataloader, "Val")

            # ---- 学习率调度 ----
            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            # ---- 日志输出（仿照HyperGKAN格式） ----
            elapsed = (time.time() - self.start_time) / 60.0
            lr = self.optimizer.param_groups[0]['lr']

            if (epoch_idx % self.log_every) == 0:
                self._logger.info('')
                self._logger.info(Colors.cyan('=' * 60))
                self._logger.info(Colors.bold(f'Epoch {epoch_idx + 1}/{self.epochs}'))
                self._logger.info(Colors.cyan('=' * 60))
                self._logger.info(f'  Train Loss:      {train_loss:.4f}')
                self._logger.info(f'  Validation Loss: {val_loss:.4f}')
                self._logger.info(Colors.yellow(f'  Val MAE   :       {val_mae:.4f}'))
                self._logger.info(Colors.yellow(f'  Val RMSE  :       {val_rmse:.4f}'))
                self._logger.info(f'  Learning Rate:   {lr:.6f}')
                self._logger.info(f'  Elapsed Time:    {elapsed:.1f} minutes')

            # ---- 模型保存 ----
            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info(Colors.green(f'  ** NEW BEST MODEL! Val Loss: {val_loss:.4f} '
                                      f'(prev: {min_val_loss:.4f})'))
                min_val_loss = val_loss
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.info(Colors.red(f'Early stopping at epoch {epoch_idx + 1}'))
                    break

            if (epoch_idx % self.log_every) == 0:
                self._logger.info(Colors.cyan('=' * 60))

        # ---- 训练结束汇总 ----
        self._logger.info('')
        self._logger.info(Colors.cyan('=' * 60))
        self._logger.info(Colors.green('Training completed!'))
        self._logger.info(Colors.cyan(f'  Best epoch: {best_epoch + 1}'))
        self._logger.info(Colors.cyan(f'  Best Val Loss: {min_val_loss:.4f}'))
        self._logger.info(Colors.yellow(f'  Best Val MAE:  {best_val_mae:.4f}'))
        self._logger.info(Colors.yellow(f'  Best Val RMSE: {best_val_rmse:.4f}'))
        total_time = (time.time() - self.start_time) / 60.0
        self._logger.info(f'  Total training time: {total_time:.1f} minutes')
        if len(train_time) > 0:
            self._logger.info(f'  Avg train time/epoch: {np.mean(train_time):.1f}s')
            self._logger.info(f'  Avg eval time/epoch:  {np.mean(eval_time):.1f}s')
        self._logger.info(Colors.cyan('=' * 60))

        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)

        # ---- 绘制损失曲线 ----
        try:
            loss_curve_path = os.path.join(
                os.path.dirname(self.cache_dir), 'loss_curve.png')
            plot_loss_curve(
                train_loss_history, val_loss_history,
                loss_curve_path,
                model_name=self.model_name,
                element=self.element)
            self._logger.info(Colors.green(f'Loss curve saved to: {loss_curve_path}'))
        except Exception as e:
            self._logger.warning(f'Failed to plot loss curve: {e}')

        return {
            'best_val_loss': min_val_loss,
            'best_val_mae': best_val_mae,
            'best_val_rmse': best_val_rmse,
            'best_epoch': best_epoch,
            'total_epochs': epoch_idx + 1,
            'total_time_min': total_time,
        }

    def _train_epoch_with_accumulation(self, train_dataloader, epoch_idx, loss_func=None):
        """
        支持梯度累积的训练epoch
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        self.optimizer.zero_grad()

        pbar = tqdm(train_dataloader,
                    desc=f'Epoch {epoch_idx + 1} [Train]',
                    leave=False)

        for i, batch in enumerate(pbar):
            batch.to_tensor(self.device)
            batch_loss = loss_func(batch)
            # 梯度累积：loss除以累积步数
            scaled_loss = batch_loss / self.accumulation_steps
            scaled_loss.backward()

            losses.append(batch_loss.item())

            if (i + 1) % self.accumulation_steps == 0:
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})

        # 处理最后不足accumulation_steps的batch
        if (len(train_dataloader)) % self.accumulation_steps != 0:
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        """
        验证epoch（带进度条）
        """
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            losses = []

            pbar = tqdm(eval_dataloader,
                        desc=f'Epoch {epoch_idx + 1} [Val]',
                        leave=False)

            for batch in pbar:
                batch.to_tensor(self.device)
                batch_loss = loss_func(batch)
                losses.append(batch_loss.item())
                pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})

            mean_loss = np.mean(losses)
            if self._writer is not None:
                self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss

    def evaluate(self, test_dataloader):
        """
        测试集评估，输出MAE和RMSE（仿照HyperGKAN日志格式）
        """
        self._logger.info('')
        self._logger.info(Colors.cyan('=' * 60))
        self._logger.info(Colors.bold('Starting evaluation on test set...'))
        self._logger.info(Colors.cyan('=' * 60))

        # 先用父类方法保存预测结果
        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []

            pbar = tqdm(test_dataloader, desc='[Test]', leave=False)
            for batch in pbar:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())

        y_preds = np.concatenate(y_preds, axis=0)
        y_truths = np.concatenate(y_truths, axis=0)

        # 计算整体指标
        mae = np.mean(np.abs(y_preds - y_truths))
        rmse = np.sqrt(np.mean((y_preds - y_truths) ** 2))

        self._logger.info('')
        self._logger.info(Colors.cyan('=' * 60))
        self._logger.info(Colors.bold(f'Test Results: {self.model_name} on {self.element}'))
        self._logger.info(Colors.cyan('=' * 60))
        self._logger.info(Colors.yellow(f'  Test MAE  : {mae:.4f}'))
        self._logger.info(Colors.yellow(f'  Test RMSE : {rmse:.4f}'))

        # 计算各时间步的指标
        n_timesteps = y_preds.shape[1]
        self._logger.info(Colors.cyan(f'  Time steps: {n_timesteps}'))
        self._logger.info(Colors.cyan('-' * 40))
        self._logger.info(Colors.bold(f'  {"Step":>4}  {"MAE":>10}  {"RMSE":>10}'))
        self._logger.info(Colors.cyan('-' * 40))

        step_results = {'MAE': [], 'RMSE': []}
        for t in range(n_timesteps):
            t_mae = np.mean(np.abs(y_preds[:, t] - y_truths[:, t]))
            t_rmse = np.sqrt(np.mean((y_preds[:, t] - y_truths[:, t]) ** 2))
            step_results['MAE'].append(t_mae)
            step_results['RMSE'].append(t_rmse)
            self._logger.info(f'  {t + 1:>4}  {t_mae:>10.4f}  {t_rmse:>10.4f}')

        self._logger.info(Colors.cyan('-' * 40))
        self._logger.info(Colors.bold(f'  {"Avg":>4}  {mae:>10.4f}  {rmse:>10.4f}'))
        self._logger.info(Colors.cyan('=' * 60))

        # 保存预测结果
        ensure_dir(self.evaluate_res_dir)
        filename = (datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' +
                    self.model_name + '_' + self.element + '_predictions.npz')
        np.savez_compressed(
            os.path.join(self.evaluate_res_dir, filename),
            prediction=y_preds, truth=y_truths)
        self._logger.info(Colors.green(f'Predictions saved to: {os.path.join(self.evaluate_res_dir, filename)}'))

        # 保存CSV格式指标
        import pandas as pd
        df = pd.DataFrame(step_results, index=range(1, n_timesteps + 1))
        csv_filename = (datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' +
                        self.model_name + '_' + self.element + '_metrics.csv')
        csv_path = os.path.join(self.evaluate_res_dir, csv_filename)
        df.to_csv(csv_path, index_label='Step')
        self._logger.info(Colors.green(f'Metrics saved to: {csv_path}'))
        self._logger.info("\n" + Colors.cyan(str(df)))

        # ---- 绘制预测可视化 (与 HyperGKAN 一致) ----
        try:
            plot_dir = os.path.dirname(self.evaluate_res_dir)
            pred_plot_path = os.path.join(
                plot_dir, f'{self.model_name}_{self.element}_predictions.png')
            pred_path, analysis_path = plot_predictions(
                pred=y_preds,
                target=y_truths,
                save_path=pred_plot_path,
                model_name=self.model_name,
                num_samples=4,
                num_stations=4,
                element=self.element)
            self._logger.info(Colors.green(f'Prediction plot saved to: {pred_path}'))
            self._logger.info(Colors.green(f'Analysis plot saved to: {analysis_path}'))
        except Exception as e:
            self._logger.warning(f'Failed to plot predictions: {e}')
            import traceback
            self._logger.warning(traceback.format_exc())
            pred_path = None
            analysis_path = None

        # 返回包含完整信息的字典
        return {
            'mae': mae,
            'rmse': rmse,
            'step_results': df,
            'n_timesteps': n_timesteps,
            'predictions_file': os.path.join(self.evaluate_res_dir, filename),
            'metrics_file': csv_path,
            'prediction_plot': pred_path if 'pred_path' in dir() else None,
            'analysis_plot': analysis_path if 'analysis_path' in dir() else None,
        }
