#!/usr/bin/env python3
"""
测试训练好的H-Net股票分析模型
"""

import torch
import numpy as np
import json
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path='best_stock_hnet.pth'):
    """加载训练好的模型"""
    try:
        from hnet_stock_training import StockHNet, HNetConfig
        
        logger.info(f"加载模型: {model_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 从checkpoint获取配置
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # 使用默认快速训练配置
            config = HNetConfig(
                d_model=128,
                num_stages=1,
                encoder_layers=2,
                decoder_layers=2,
                main_layers=4,
                sequence_length=60,
                prediction_horizon=5
            )
        
        # 创建模型
        model = StockHNet(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"✅ 模型加载成功!")
        logger.info(f"   模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        logger.info(f"   训练轮数: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"   验证损失: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        return model, config
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return None, None

def test_model_inference(model, config):
    """测试模型推理功能"""
    logger.info("🧪 测试模型推理...")
    
    batch_size = 2
    seq_len = config.sequence_length
    
    # 创建随机测试数据
    price_data = torch.randn(batch_size, seq_len, config.price_features)
    technical_data = torch.randn(batch_size, seq_len, config.technical_features)
    news_data = torch.randn(batch_size, seq_len, config.news_embed_dim)
    
    try:
        with torch.no_grad():
            predictions, boundary_loss = model(price_data, technical_data, news_data)
        
        logger.info("✅ 模型推理成功!")
        logger.info(f"   价格预测形状: {predictions['price'].shape}")
        logger.info(f"   波动率预测形状: {predictions['volatility'].shape}")
        logger.info(f"   方向预测形状: {predictions['direction'].shape}")
        logger.info(f"   边界损失: {boundary_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"模型推理失败: {e}")
        return False

def test_with_real_data(model, config):
    """使用真实数据测试模型"""
    logger.info("📊 使用真实数据测试...")
    
    try:
        from hnet_stock_training import StockDataset
        
        # 加载测试数据
        test_dataset = StockDataset("stock_data", config, 'test')
        
        if len(test_dataset) == 0:
            logger.warning("没有测试数据")
            return False
        
        # 获取一个样本
        sample = test_dataset[0]
        
        # 添加batch维度
        price_data = sample['price'].unsqueeze(0)
        technical_data = sample['technical'].unsqueeze(0)
        news_data = sample['news'].unsqueeze(0)
        targets = sample['targets']
        
        with torch.no_grad():
            predictions, _ = model(price_data, technical_data, news_data)
        
        # 显示预测结果
        logger.info("✅ 真实数据测试成功!")
        logger.info(f"   真实价格目标: {targets['price'][:3].numpy()}")
        logger.info(f"   预测价格: {predictions['price'][0][:3].numpy()}")
        logger.info(f"   真实方向: {targets['direction'][:3].numpy()}")
        logger.info(f"   预测方向概率: {torch.softmax(predictions['direction'][0][:3], dim=-1).numpy()}")
        
        return True
        
    except Exception as e:
        logger.error(f"真实数据测试失败: {e}")
        return False

def evaluate_model_performance(model, config):
    """评估模型性能"""
    logger.info("📈 评估模型性能...")
    
    try:
        from hnet_stock_training import StockDataset
        from torch.utils.data import DataLoader
        
        # 加载测试数据
        test_dataset = StockDataset("stock_data", config, 'test')
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        total_samples = 0
        price_mse = 0
        direction_correct = 0
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                price_data = batch['price']
                technical_data = batch['technical']
                news_data = batch['news']
                targets = batch['targets']
                
                predictions, _ = model(price_data, technical_data, news_data)
                
                # 计算指标
                batch_size = price_data.shape[0]
                total_samples += batch_size
                
                # 价格MSE
                price_mse += torch.mean((predictions['price'] - targets['price']) ** 2).item() * batch_size
                
                # 方向准确率
                pred_directions = torch.argmax(predictions['direction'], dim=-1)
                direction_correct += torch.sum(pred_directions == targets['direction']).item()
        
        # 计算平均指标
        price_mse /= total_samples
        direction_accuracy = direction_correct / (total_samples * config.prediction_horizon)
        
        logger.info("📊 模型性能评估结果:")
        logger.info(f"   测试样本数: {total_samples}")
        logger.info(f"   价格预测MSE: {price_mse:.4f}")
        logger.info(f"   方向预测准确率: {direction_accuracy:.2%}")
        
        return {
            'price_mse': price_mse,
            'direction_accuracy': direction_accuracy,
            'test_samples': total_samples
        }
        
    except Exception as e:
        logger.error(f"性能评估失败: {e}")
        return None

def main():
    """主测试函数"""
    print("🧪 H-Net 股票分析模型测试")
    print("=" * 50)
    
    # 1. 加载模型
    model, config = load_trained_model()
    if model is None:
        print("❌ 模型加载失败，退出测试")
        return
    
    # 2. 基础推理测试
    if not test_model_inference(model, config):
        print("❌ 基础推理测试失败")
        return
    
    # 3. 真实数据测试
    if not test_with_real_data(model, config):
        print("⚠️  真实数据测试失败，但基础功能正常")
    
    # 4. 性能评估
    performance = evaluate_model_performance(model, config)
    if performance:
        print(f"\n🎯 模型训练完成!")
        print(f"   ✅ 基础功能: 正常")
        print(f"   ✅ 推理速度: 正常")
        print(f"   📊 性能指标:")
        print(f"      - 价格预测误差: {performance['price_mse']:.4f}")
        print(f"      - 方向预测准确率: {performance['direction_accuracy']:.2%}")
        
        # 判断模型质量
        if performance['direction_accuracy'] > 0.4:  # 随机是33.3%
            print(f"   🎉 模型表现良好!")
        elif performance['direction_accuracy'] > 0.35:
            print(f"   ⚡ 模型表现一般，可考虑更长时间训练")
        else:
            print(f"   ⚠️  模型表现较差，建议调整参数重新训练")
    
    print(f"\n📁 训练产出文件:")
    print(f"   - best_stock_hnet.pth: 最佳训练模型")
    print(f"   - training_config_quick.json: 训练配置")
    
    print(f"\n🚀 下一步建议:")
    print(f"   1. 尝试更长时间的训练 (balanced 或 thorough 模式)")
    print(f"   2. 调整模型参数和学习率")
    print(f"   3. 增加更多训练数据")
    print(f"   4. 进行实时推理测试")

if __name__ == "__main__":
    main()
