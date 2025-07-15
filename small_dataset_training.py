#!/usr/bin/env python3
"""
针对小数据集的优化训练配置 - 增强版本，包含详细训练监控
"""

import torch
import torch.nn as nn
from hnet_stock_training import HNetConfig, StockTrainer, StockDataset
import json
import os
import time
import sys
from datetime import datetime, timedelta
from collections import defaultdict, deque

class SmallDatasetConfig:
    """针对小数据集的优化配置"""
    
    @staticmethod
    def create_optimized_configs():
        """创建针对小数据集优化的训练配置"""
        
        configs = {
            'nano': {
                'name': 'Nano - 极小模型 (防过拟合)',
                'd_model': 64,
                'num_stages': 1,
                'encoder_layers': 2,
                'decoder_layers': 2,
                'main_layers': 4,
                'chunk_ratios': [2],
                'batch_size': 16,
                'learning_rate': 1e-3,
                'max_epochs': 8,
                'dropout': 0.3,
                'weight_decay': 1e-3,
                'params_estimate': '0.3M',
                'description': '最小模型，适合647个样本'
            },
            
            'micro': {
                'name': 'Micro - 微型模型',
                'd_model': 96,
                'num_stages': 1,
                'encoder_layers': 2,
                'decoder_layers': 2,
                'main_layers': 6,
                'chunk_ratios': [3],
                'batch_size': 12,
                'learning_rate': 8e-4,
                'max_epochs': 12,
                'dropout': 0.25,
                'weight_decay': 5e-4,
                'params_estimate': '0.6M',
                'description': '微型模型，保守训练'
            },
            
            'small_regularized': {
                'name': 'Small Regularized - 小模型+强正则化',
                'd_model': 128,
                'num_stages': 1,
                'encoder_layers': 3,
                'decoder_layers': 3,
                'main_layers': 6,
                'chunk_ratios': [4],
                'batch_size': 8,
                'learning_rate': 5e-4,
                'max_epochs': 15,
                'dropout': 0.4,
                'weight_decay': 1e-3,
                'params_estimate': '0.9M',
                'description': '小模型配合强正则化'
            },
            
            'early_stop': {
                'name': 'Early Stop - 早停策略',
                'd_model': 128,
                'num_stages': 2,
                'encoder_layers': 3,
                'decoder_layers': 3,
                'main_layers': 8,
                'chunk_ratios': [4, 3],
                'batch_size': 8,
                'learning_rate': 3e-4,
                'max_epochs': 50,  # 高epoch但会早停
                'dropout': 0.35,
                'weight_decay': 8e-4,
                'early_stopping': True,
                'patience': 5,
                'params_estimate': '1.2M',
                'description': '使用早停避免过拟合'
            }
        }
        
        return configs

class TrainingMonitor:
    """训练监控器 - 提供详细的训练过程可视化"""
    
    def __init__(self):
        self.epoch_start_time = None
        self.batch_start_time = None
        self.loss_history = defaultdict(list)
        self.speed_history = deque(maxlen=50)  # 保留最近50个批次的速度
        self.total_batches = 0
        self.current_epoch = 0
        
    def start_epoch(self, epoch, total_batches):
        """开始新的epoch"""
        self.current_epoch = epoch
        self.total_batches = total_batches
        self.epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🚀 Epoch {epoch+1} 开始 ({datetime.now().strftime('%H:%M:%S')})")
        print(f"{'='*60}")
        
    def start_batch(self, batch_idx):
        """开始新的批次"""
        self.batch_start_time = time.time()
        
    def end_batch(self, batch_idx, losses, batch_size):
        """结束批次，记录信息"""
        if self.batch_start_time is None:
            return
            
        batch_time = time.time() - self.batch_start_time
        samples_per_sec = batch_size / batch_time if batch_time > 0 else 0
        self.speed_history.append(samples_per_sec)
        
        # 记录损失
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                self.loss_history[key].append(value.item())
            else:
                self.loss_history[key].append(float(value))
        
        # 计算进度
        progress = (batch_idx + 1) / self.total_batches
        
        # 计算预估剩余时间
        if len(self.speed_history) > 0:
            avg_speed = sum(self.speed_history) / len(self.speed_history)
            remaining_batches = self.total_batches - (batch_idx + 1)
            remaining_time = remaining_batches * batch_time if avg_speed > 0 else 0
            eta = datetime.now() + timedelta(seconds=remaining_time)
        else:
            eta = None
            avg_speed = 0
        
        # 打印详细信息
        if (batch_idx + 1) % max(1, self.total_batches // 10) == 0 or batch_idx == 0:
            self._print_batch_info(batch_idx, progress, losses, samples_per_sec, eta)
    
    def _print_batch_info(self, batch_idx, progress, losses, speed, eta):
        """打印批次信息"""
        # 进度条
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '▓' * (bar_length - filled_length)
        
        # 主要损失
        total_loss = losses.get('total', 0)
        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss.item()
        
        print(f"\r批次 {batch_idx+1:4d}/{self.total_batches} |{bar}| "
              f"{progress*100:5.1f}% | "
              f"损失: {total_loss:.4f} | "
              f"速度: {speed:.1f} samples/s", end="")
        
        # 每20个批次详细输出
        if (batch_idx + 1) % max(1, self.total_batches // 5) == 0:
            print()  # 换行
            self._print_detailed_losses(losses)
            if eta:
                print(f"   ⏰ 预计完成: {eta.strftime('%H:%M:%S')}")
                print(f"   🏃 当前速度: {speed:.1f} samples/s")
    
    def _print_detailed_losses(self, losses):
        """打印详细损失分解"""
        print("   📊 损失分解:", end="")
        for key, value in losses.items():
            if key != 'total':
                if isinstance(value, torch.Tensor):
                    value = value.item()
                print(f" {key}={value:.4f}", end="")
        print()
    
    def end_epoch(self, train_loss, val_losses):
        """结束epoch"""
        if self.epoch_start_time is None:
            return
            
        epoch_time = time.time() - self.epoch_start_time
        
        print(f"\n{'='*60}")
        print(f"✅ Epoch {self.current_epoch+1} 完成")
        print(f"   ⏱️  用时: {epoch_time:.2f}秒")
        print(f"   📈 训练损失: {train_loss:.6f}")
        
        # 验证损失详情
        if isinstance(val_losses, dict):
            print(f"   📉 验证损失: {val_losses.get('total', 0):.6f}")
            for key, value in val_losses.items():
                if key != 'total':
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    print(f"      - {key}: {value:.6f}")
        else:
            print(f"   📉 验证损失: {val_losses:.6f}")
        
        # GPU内存使用
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2
            print(f"   💾 GPU内存: {memory_used:.1f}MB 已用, {memory_cached:.1f}MB 缓存")
        
        print(f"{'='*60}")

class EarlyStoppingCallback:
    """早停回调"""
    
    def __init__(self, patience=5, min_delta=0.001, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class RegularizedStockTrainer(StockTrainer):
    """增强正则化的训练器"""
    
    def __init__(self, config, device='cpu', use_early_stopping=False):
        super().__init__(config, device)
        
        self.use_early_stopping = use_early_stopping
        if use_early_stopping:
            self.early_stopping = EarlyStoppingCallback(
                patience=getattr(config, 'patience', 5),
                min_delta=0.001
            )
        
        # 增强正则化
        self.setup_regularization(config)
    
    def setup_regularization(self, config):
        """设置正则化"""
        # 更强的权重衰减
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = config.weight_decay
        
        # Dropout调度器
        self.dropout_scheduler = self.create_dropout_scheduler(config.dropout)
    
    def create_dropout_scheduler(self, max_dropout):
        """创建动态dropout调度器"""
        def scheduler(epoch):
            # 随着训练进行逐渐增加dropout
            return min(max_dropout, 0.1 + (max_dropout - 0.1) * epoch / 20)
        return scheduler
    
    def apply_dropout_schedule(self, epoch):
        """应用dropout调度"""
        current_dropout = self.dropout_scheduler(epoch)
        
        # 更新模型中的dropout
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = current_dropout
    
    def compute_l2_regularization(self):
        """计算L2正则化"""
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param, 2) ** 2
        return l2_reg * self.config.weight_decay
    
    def train_epoch(self, dataloader):
        """训练一个epoch，增强正则化和详细监控"""
        self.model.train()
        
        # 初始化监控
        if not hasattr(self, 'monitor'):
            self.monitor = TrainingMonitor()
        
        total_losses = {'total': 0, 'price': 0, 'volatility': 0, 'direction': 0, 'boundary': 0, 'l2_reg': 0}
        num_batches = len(dataloader)
        
        # 开始epoch监控
        current_epoch = getattr(self, 'current_training_epoch', 0)
        self.monitor.start_epoch(current_epoch, num_batches)
        
        for batch_idx, batch in enumerate(dataloader):
            # 开始批次监控
            self.monitor.start_batch(batch_idx)
            
            # 数据到设备
            price_data = batch['price'].to(self.device)
            technical_data = batch['technical'].to(self.device)
            news_data = batch['news'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions, boundary_loss = self.model(price_data, technical_data, news_data)
            
            # 计算损失
            losses = self.compute_loss(predictions, targets, boundary_loss)
            
            # 添加L2正则化
            l2_reg = self.compute_l2_regularization()
            losses['total'] = losses['total'] + l2_reg
            losses['l2_reg'] = l2_reg
            
            # 反向传播
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录损失
            batch_losses = {k: v.item() for k, v in losses.items()}
            
            for k, v in batch_losses.items():
                if k in total_losses:
                    total_losses[k] += v
            
            # 结束批次监控
            self.monitor.end_batch(batch_idx, batch_losses, len(price_data))
        
        # 计算平均损失
        avg_losses = {}
        for key in total_losses:
            avg_losses[key] = total_losses[key] / num_batches
        
        return avg_losses
    
    def train(self, train_dataset, val_dataset):
        """训练模型，支持早停"""
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"开始训练，最大epoch: {self.config.max_epochs}")
        if self.use_early_stopping:
            print(f"使用早停机制，耐心度: {self.early_stopping.patience}")
        
        for epoch in range(self.config.max_epochs):
            # 设置当前epoch用于监控
            self.current_training_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            train_loss = train_losses['total']
            
            # 验证
            val_losses = self.validate(val_loader)
            val_loss = val_losses['total']
            
            # 结束epoch监控
            if hasattr(self, 'monitor'):
                self.monitor.end_epoch(train_loss, val_losses)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            if self.use_early_stopping:
                if self.early_stopping(val_loss, self.model):
                    print(f"Early stopping at epoch {epoch+1}")
                    self.early_stopping.restore_best_weights(self.model)
                    break
            
            # 保存最佳模型
            if val_loss < getattr(self, 'best_val_loss', float('inf')):
                self.best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config,
                    'epoch': epoch
                }, 'best_small_dataset_model.pth')

class ExtendedHNetConfig(HNetConfig):
    """扩展的配置类，支持早停等功能"""
    def __init__(self, **kwargs):
        super().__init__()
        # 设置默认的早停相关属性
        self.early_stopping = False
        self.patience = 5
        
        # 应用传入的配置
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_small_data_trainer(mode='nano', data_dir='stock_data_eodhd_extended'):
    """创建针对小数据集的训练器"""
    
    configs = SmallDatasetConfig.create_optimized_configs()
    
    if mode not in configs:
        print(f"未知模式: {mode}")
        print(f"可用模式: {list(configs.keys())}")
        return None, None
    
    config_dict = configs[mode]
    
    # 创建扩展的HNetConfig
    config = ExtendedHNetConfig(
        d_model=config_dict['d_model'],
        num_stages=config_dict['num_stages'],
        encoder_layers=config_dict['encoder_layers'],
        decoder_layers=config_dict['decoder_layers'],
        main_layers=config_dict['main_layers'],
        chunk_ratios=config_dict['chunk_ratios'],
        batch_size=config_dict['batch_size'],
        learning_rate=config_dict['learning_rate'],
        max_epochs=config_dict['max_epochs'],
        dropout=config_dict['dropout'],
        weight_decay=config_dict['weight_decay'],
        sequence_length=60,
        prediction_horizon=5
    )
    
    # 添加额外配置
    if 'early_stopping' in config_dict:
        config.early_stopping = config_dict['early_stopping']
        config.patience = config_dict.get('patience', 5)
    
    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_early_stopping = getattr(config, 'early_stopping', False)
    
    trainer = RegularizedStockTrainer(config, device, use_early_stopping)
    
    print(f"✅ 创建 {config_dict['name']} 训练器")
    print(f"   模型参数: {config_dict['params_estimate']}")
    print(f"   描述: {config_dict['description']}")
    
    return trainer, config

def main():
    """主函数"""
    print("🎯 小数据集优化训练解决方案")
    print("=" * 50)
    
    # 显示配置选项
    configs = SmallDatasetConfig.create_optimized_configs()
    
    print("📋 可用的小数据集优化配置:")
    for i, (key, config) in enumerate(configs.items(), 1):
        print(f"  {i}. {key}: {config['name']}")
        print(f"     - {config['description']}")
        print(f"     - 参数量: {config['params_estimate']}")
        print(f"     - Epoch: {config['max_epochs']}")
        print()
    
    # 用户选择
    try:
        choice = input("请选择配置 (1-4): ").strip()
        mode_keys = list(configs.keys())
        mode = mode_keys[int(choice) - 1]
    except (ValueError, IndexError):
        print("无效选择，使用默认nano模式")
        mode = 'nano'
    
    print(f"\n🚀 使用 {mode} 配置进行训练")
    
    # 检查数据
    data_options = []
    if os.path.exists("stock_data_eodhd_extended"):
        data_options.append(("stock_data_eodhd_extended", "EODHD扩展数据"))
    if os.path.exists("augmented_data"):
        data_options.append(("augmented_data", "增强数据"))
    
    if not data_options:
        print("❌ 未找到训练数据")
        print("请先运行数据预处理或数据增强")
        return
    
    print(f"\n📂 可用数据:")
    for i, (path, desc) in enumerate(data_options, 1):
        print(f"  {i}. {desc} ({path})")
    
    try:
        data_choice = input("请选择数据 (1-2): ").strip()
        data_dir = data_options[int(data_choice) - 1][0]
    except (ValueError, IndexError):
        data_dir = data_options[0][0]
    
    print(f"使用数据: {data_dir}")
    
    # 创建训练器
    trainer, config = create_small_data_trainer(mode, data_dir)
    
    if trainer is None or config is None:
        print("❌ 创建训练器失败")
        return
    
    try:
        # 创建数据集
        train_dataset = StockDataset(data_dir, config, 'train')
        val_dataset = StockDataset(data_dir, config, 'val')
        
        print(f"\n📊 数据集信息:")
        print(f"   训练样本: {len(train_dataset)}")
        print(f"   验证样本: {len(val_dataset)}")
        
        # 计算过拟合风险
        params_count = sum(p.numel() for p in trainer.model.parameters())
        samples_per_param = len(train_dataset) / params_count
        
        print(f"   模型参数: {params_count:,}")
        print(f"   每参数样本数: {samples_per_param:.6f}")
        
        if samples_per_param < 0.001:
            print("   ⚠️  过拟合风险: 🔴 高")
        else:
            print("   ✅ 过拟合风险: 🟢 可控")
        
        # 开始训练
        response = input(f"\n开始训练? (Y/n): ").strip().lower()
        if response != 'n':
            trainer.train(train_dataset, val_dataset)
            print(f"\n✅ 训练完成!")
            print(f"模型已保存为: best_small_dataset_model.pth")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
