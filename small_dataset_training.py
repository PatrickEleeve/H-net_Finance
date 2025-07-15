#!/usr/bin/env python3
"""
针对小数据集的优化训练配置
"""

import torch
import torch.nn as nn
from hnet_stock_training import HNetConfig, StockTrainer, StockDataset
import json
import os

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
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch，增强正则化"""
        self.model.train()
        
        # 应用dropout调度
        self.apply_dropout_schedule(epoch)
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据到设备
            price_data = batch['price'].to(self.device)
            technical_data = batch['technical'].to(self.device)
            news_data = batch['news'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(price_data, technical_data, news_data)
            
            # 计算损失
            losses = self.compute_losses(outputs, targets)
            total_loss_batch = sum(losses.values())
            
            # 添加L2正则化
            l2_reg = self.compute_l2_regularization()
            total_loss_batch += l2_reg
            
            # 反向传播
            total_loss_batch.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def compute_l2_regularization(self):
        """计算L2正则化"""
        l2_reg = 0
        for param in self.model.parameters():
            l2_reg += torch.norm(param, 2) ** 2
        return l2_reg * self.config.weight_decay
    
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
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_losses = self.validate(val_loader)
            val_loss = val_losses['total']
            
            print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
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

def create_small_data_trainer(mode='nano', data_dir='stock_data_eodhd_extended'):
    """创建针对小数据集的训练器"""
    
    configs = SmallDatasetConfig.create_optimized_configs()
    
    if mode not in configs:
        print(f"未知模式: {mode}")
        print(f"可用模式: {list(configs.keys())}")
        return None
    
    config_dict = configs[mode]
    
    # 创建HNetConfig
    config = HNetConfig(
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
    
    if trainer is None:
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
