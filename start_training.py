#!/usr/bin/env python3
"""
启动H-Net股票分析模型训练的简易脚本
"""

import os
import sys
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_data_availability():
    """检查训练数据是否可用"""
    data_dir = "stock_data"
    required_splits = ['train', 'val', 'test']
    required_files = ['merged_dataset_price.npy', 'merged_dataset_technical.npy', 
                     'merged_dataset_news.npy', 'merged_dataset_targets.npz']
    
    logger.info("🔍 检查训练数据...")
    
    if not os.path.exists(data_dir):
        logger.error(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    # 检查元数据文件
    metadata_file = os.path.join(data_dir, "merged_dataset_metadata.json")
    if not os.path.exists(metadata_file):
        logger.error("❌ 找不到数据集元数据文件")
        return False
    
    # 检查训练、验证、测试数据
    for split in required_splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            logger.error(f"❌ 找不到{split}数据目录: {split_dir}")
            return False
        
        for file in required_files:
            file_path = os.path.join(split_dir, file)
            if not os.path.exists(file_path):
                logger.error(f"❌ 找不到数据文件: {file_path}")
                return False
    
    logger.info("✅ 所有训练数据文件检查通过")
    return True

def check_dependencies():
    """检查必要的依赖是否安装"""
    logger.info("🔍 检查依赖...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ 缺少必要的依赖包: {missing_packages}")
        logger.info("请运行: pip install torch numpy pandas tqdm")
        return False
    
    logger.info("✅ 所有依赖检查通过")
    return True

def estimate_training_time():
    """估算训练时间"""
    import json
    
    metadata_file = "stock_data/merged_dataset_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    train_samples = metadata['train_samples']
    sequence_length = metadata['sequence_length']
    
    # 简单估算（基于经验）
    estimated_minutes = (train_samples * sequence_length) / 10000  # 粗略估算
    
    logger.info(f"📊 训练数据统计:")
    logger.info(f"   训练样本: {train_samples}")
    logger.info(f"   序列长度: {sequence_length}")
    logger.info(f"   预测长度: {metadata['prediction_horizon']}")
    logger.info(f"⏱️  预估训练时间: {estimated_minutes:.1f} 分钟 (CPU)")
    
    return estimated_minutes

def main():
    """主函数"""
    print("🚀 H-Net 股票分析模型训练启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装必要的包")
        sys.exit(1)
    
    # 检查数据
    if not check_data_availability():
        print("\n❌ 数据检查失败")
        print("💡 请先运行数据预处理脚本生成训练数据:")
        print("   python hnet_data_preprocess.py")
        sys.exit(1)
    
    # 估算训练时间
    estimated_time = estimate_training_time()
    
    # 用户确认
    print(f"\n⚠️  训练准备就绪，预估需要 {estimated_time:.1f} 分钟")
    response = input("是否开始训练? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ 训练取消")
        sys.exit(0)
    
    print("\n🎯 开始训练...")
    
    try:
        # 导入并运行训练脚本
        from hnet_stock_training import main as train_main
        train_main()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  训练被用户中断")
    except Exception as e:
        logger.error(f"\n❌ 训练过程中出现错误: {e}")
        logger.exception("详细错误信息:")
        sys.exit(1)
    
    print("\n✅ 训练完成!")
    print("📁 检查以下文件:")
    print("   - best_stock_hnet.pth (最佳模型)")
    print("   - training_*.log (训练日志)")

if __name__ == "__main__":
    main()
